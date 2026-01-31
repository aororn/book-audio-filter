"""
Morpho Rules v1.3 — Морфологические правила фильтрации.

Единый модуль для фильтрации ложных ошибок транскрипции.
Заменяет smart_rules.py и learned_rules.py.

ПРИНЦИП: Консервативная фильтрация.
- Фильтруем ТОЛЬКО если 100% уверены, что это ошибка Яндекса
- При любом сомнении — НЕ фильтруем (пусть человек проверит)
- Грамматическое различие (число, падеж, часть речи) = НЕ фильтровать

На основе анализа 70 golden ошибок:
- SAME_LEMMA + DIFF_NUM:  7 реальных ошибок → НЕ фильтровать
- SAME_LEMMA + DIFF_CASE: 10 реальных ошибок → НЕ фильтровать
- SAME_LEMMA + DIFF_POS:  6 реальных ошибок → НЕ фильтровать
- SAME_LEMMA + DIFF_TENSE: 2 реальных ошибки → НЕ фильтровать

ВАЖНО: Запускать через venv с установленным пакетом (pip install -e .)

v1.3 (2026-01-31): Относительный импорт из Инструменты.morphology
v1.2 (2026-01-30): proper_name фильтр теперь проверяет схожесть слов (≥50%)
v1.1 (2026-01-29): Исправлен баг с _is_proper_name()
v1.0 (2026-01-26): Начальная версия
"""

VERSION = '1.4.0'  # v1.4: Исправлен импорт morphology через sys.path
VERSION_DATE = '2026-01-31'

import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# Добавляем путь к Инструменты в sys.path
_SCRIPT_DIR = Path(__file__).parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from morphology import (
    normalize_word, get_lemma, get_pos, get_number, get_case,
    HAS_PYMORPHY, morph
)
HAS_MORPHOLOGY = HAS_PYMORPHY  # Синоним для совместимости

from .comparison import levenshtein_distance, phonetic_normalize


@dataclass
class FilterResult:
    """Результат проверки правила."""
    should_filter: bool  # True = фильтровать (ложная ошибка)
    rule_name: str
    confidence: float
    reason: str = ""


class MorphoRules:
    """
    Морфологические правила фильтрации.

    Консервативный подход: фильтруем только очевидные ошибки Яндекса.
    """

    def check(self, w1: str, w2: str) -> Optional[FilterResult]:
        """
        Проверяет пару слов.

        Args:
            w1: первое слово (транскрипт)
            w2: второе слово (оригинал)

        Returns:
            FilterResult если можно принять решение, иначе None
        """
        w1 = normalize_word(w1)
        w2 = normalize_word(w2)

        # Идентичные слова — фильтруем
        if w1 == w2:
            return FilterResult(True, 'identical', 1.0, 'Слова идентичны')

        if not HAS_MORPHOLOGY:
            return None

        # Получаем морфологию
        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
        pos1, pos2 = get_pos(w1), get_pos(w2)
        num1, num2 = get_number(w1), get_number(w2)
        case1, case2 = get_case(w1), get_case(w2)

        # =================================================================
        # ПРАВИЛО 1: Разные леммы — НЕ фильтровать
        # Это разные слова, возможно реальная ошибка
        # =================================================================
        if lemma1 != lemma2:
            # Исключение: имена собственные (Яндекс не знает имён)
            # v1.2: Добавлена проверка схожести — если слова совершенно разные
            # (например, "Рутгош" vs "лет"), это НЕ ошибка распознавания имени
            if self._is_proper_name(w1) or self._is_proper_name(w2):
                # Проверяем схожесть слов
                max_len = max(len(w1), len(w2))
                if max_len > 0:
                    dist = levenshtein_distance(w1.lower(), w2.lower())
                    similarity = 1 - (dist / max_len)
                    # Фильтруем только если слова похожи (>50% схожести)
                    if similarity >= 0.5:
                        return FilterResult(True, 'proper_name', 1.0,
                                          f'Имя собственное: {w1}/{w2}')

            # Фонетически идентичные с РАЗНЫМИ леммами — омофоны
            # НО только если одинаковая часть речи!
            if pos1 == pos2 and phonetic_normalize(w1) == phonetic_normalize(w2):
                # Проверяем: не разные ли грамматические формы
                # Если число или падеж разные — это может быть реальная ошибка
                return FilterResult(True, 'homophone', 0.95,
                                  f'Омофоны: {w1}/{w2}')

            return None  # Разные слова — пусть человек проверит

        # =================================================================
        # Далее: ОДИНАКОВАЯ лемма
        # =================================================================

        # =================================================================
        # ПРАВИЛО 2: Разная часть речи — НЕ фильтровать
        # VERB↔GRND (теряю/теряя) — реальная ошибка чтеца!
        # =================================================================
        if pos1 != pos2:
            return None  # Не фильтруем

        # =================================================================
        # ПРАВИЛО 3: Разное число — НЕ фильтровать
        # сотни/сотня, будет/будут — реальные ошибки!
        # =================================================================
        if num1 and num2 and num1 != num2:
            return None  # Не фильтруем

        # =================================================================
        # ПРАВИЛО 4: Разный падеж — НЕ фильтровать
        # преграды/преград, награда/награды — реальные ошибки!
        # =================================================================
        if case1 and case2 and case1 != case2:
            return None  # Не фильтруем

        # =================================================================
        # ПРАВИЛО 5: Разное время глагола — НЕ фильтровать
        # получится/получилось — реальная ошибка!
        # =================================================================
        if pos1 in ('VERB', 'INFN'):
            tense1 = self._get_tense(w1)
            tense2 = self._get_tense(w2)
            if tense1 and tense2 and tense1 != tense2:
                return None  # Не фильтруем

        # =================================================================
        # ПРАВИЛО 6: Приставка по- — НЕ фильтровать
        # больше/побольше — реальная ошибка!
        # =================================================================
        if w1.startswith('по') and len(w1) > 3 and w1[2:] == w2:
            return None
        if w2.startswith('по') and len(w2) > 3 and w2[2:] == w1:
            return None

        # =================================================================
        # ЕСЛИ ДОШЛИ СЮДА: одинаковая лемма, одинаковые грамм. признаки
        # Это ложная ошибка Яндекса — фильтруем
        # =================================================================
        return FilterResult(True, 'same_form', 0.95,
                          f'Одинаковая форма: {lemma1}')

    def _is_proper_name(self, word: str) -> bool:
        """
        Проверяет, является ли слово именем собственным.

        ВАЖНО: pymorphy3 иногда ошибочно определяет русские слова как имена.
        Например, "когдато" (слитное "когда-то") определяется как имя Name.

        Поэтому добавляем дополнительные проверки:
        1. score должен быть достаточно высоким (>0.5)
        2. Слово не должно быть частью составного слова (то, либо, нибудь)
        """
        if not HAS_PYMORPHY or not morph:
            return False

        # Исключаем русские составные слова (когда-то → когдато)
        compound_suffixes = ('то', 'либо', 'нибудь', 'таки', 'ка', 'де', 'мол')
        word_lower = word.lower()
        for suffix in compound_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                # Слово вроде "когдато", "кудато", "чтолибо" — не имя
                return False

        parsed = morph.parse(word)
        if parsed:
            p = parsed[0]
            tag_str = str(p.tag)
            is_name = 'Name' in tag_str or 'Surn' in tag_str

            # Дополнительная проверка: score должен быть высоким
            # и слово должно быть из словаря (не "угадано")
            if is_name and p.score < 0.5:
                return False

            return is_name
        return False

    def _get_tense(self, word: str) -> Optional[str]:
        """Получает время глагола."""
        if not HAS_PYMORPHY or not morph:
            return None

        parsed = morph.parse(word)
        if parsed:
            tag = parsed[0].tag
            if 'past' in tag:
                return 'past'
            elif 'pres' in tag:
                return 'pres'
            elif 'futr' in tag:
                return 'futr'
        return None


# Singleton
_morpho_rules_instance = None

def get_morpho_rules() -> MorphoRules:
    """Возвращает глобальный экземпляр MorphoRules."""
    global _morpho_rules_instance
    if _morpho_rules_instance is None:
        _morpho_rules_instance = MorphoRules()
    return _morpho_rules_instance


def is_morpho_false_positive(w1: str, w2: str) -> Tuple[bool, str]:
    """
    Быстрая проверка через морфологические правила.

    Returns:
        (should_filter, rule_name)
    """
    result = get_morpho_rules().check(w1, w2)
    if result:
        return result.should_filter, result.rule_name
    return False, 'unknown'


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_golden_errors():
    """Тестирует на golden ошибках — ни одна не должна фильтроваться."""
    import json
    from pathlib import Path

    rules = get_morpho_rules()

    # Golden substitution ошибки (из анализа)
    golden_pairs = [
        # SAME_LEMMA_DIFF_NUM
        ('сотни', 'сотня'),
        ('предводителя', 'предводителей'),
        ('руки', 'руку'),
        ('будет', 'будут'),
        # SAME_LEMMA_DIFF_CASE
        ('преград', 'преграды'),
        ('награды', 'награда'),
        ('господин', 'господином'),
        ('цвету', 'цвет'),
        ('указ', 'указу'),
        # SAME_LEMMA_DIFF_POS (VERB↔GRND)
        ('теряю', 'теряя'),
        ('продолжая', 'продолжал'),
        ('добавила', 'добавив'),
        ('услышал', 'услышав'),
        # SAME_LEMMA_DIFF_TENSE
        ('получится', 'получилось'),
        # SAME_LEMMA приставка по-
        ('побольше', 'больше'),
    ]

    print('='*60)
    print('ТЕСТ: Golden ошибки НЕ должны фильтроваться')
    print('='*60)

    errors = 0
    for w1, w2 in golden_pairs:
        result = rules.check(w1, w2)
        if result and result.should_filter:
            print(f'FAIL: {w1} -> {w2} отфильтровано как {result.rule_name}')
            errors += 1
        else:
            print(f'OK: {w1} -> {w2} НЕ фильтруется')

    print()
    print(f'Результат: {len(golden_pairs) - errors}/{len(golden_pairs)} OK')
    return errors == 0


if __name__ == '__main__':
    test_golden_errors()
