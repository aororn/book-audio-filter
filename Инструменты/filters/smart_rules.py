"""
Smart Rules v2.0 — Умные правила фильтрации на основе морфологии.

Заменяют статические списки пар на алгоритмическую проверку,
что позволяет обобщать на новые данные без ручного добавления.

v2.0: Теперь smart_rules ЗАМЕНЯЮТ старые правила:
- same_lemma → smart_lemma
- homophone → smart_homophone
- levenshtein_similar_lemma → smart_levenshtein
- grammar_ending → smart_grammar

Основные правила:
- smart_lemma — одинаковая лемма (кроме опасных пар)
- smart_homophone — фонетическая идентичность
- smart_levenshtein — морфологически близкие слова
- smart_grammar — грамматические вариации
- smart_aspect_pair — видовые пары глаголов
- smart_reflexive — возвратные формы (-ся/-сь)
- smart_participle — причастия и их формы

v1.1 (2026-01-26): Исправления фонетики
v1.0 (2026-01-26): Начальная версия
"""

VERSION = '2.0.0'
VERSION_DATE = '2026-01-26'

import re
from functools import lru_cache
from typing import Optional, Tuple, Dict, Set, List
from dataclasses import dataclass

# Импорт морфологии
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from morphology import (
        normalize_word,
        get_lemma,
        get_pos,
        get_number,
        get_gender,
        get_case,
        get_aspect,
        get_voice,
        get_tense,
        get_verb_info,
        is_aspect_pair,
        is_same_verb_base,
        HAS_PYMORPHY,
        morph,
    )
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False
    HAS_PYMORPHY = False
    morph = None

# Импорт Левенштейна
try:
    from .comparison import levenshtein_distance, levenshtein_ratio
except ImportError:
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    def levenshtein_ratio(s1: str, s2: str) -> int:
        dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return int((1 - dist / max_len) * 100) if max_len else 100


# =============================================================================
# ФОНЕТИЧЕСКИЕ ПАТТЕРНЫ ДЛЯ РУССКОГО ЯЗЫКА
# =============================================================================

PHONETIC_REPLACEMENTS = [
    # Гласные
    (r'о', 'а'),      # безударное о → а
    (r'е', 'и'),      # безударное е → и
    (r'я', 'и'),      # безударное я → и (после согласных)
    # Согласные
    (r'тс', 'ц'),     # -тся/-ться → ца
    (r'тьс', 'ц'),
    (r'дс', 'ц'),     # подсказка → поцказка
    (r'чн', 'шн'),    # конечно → конешно
    (r'чт', 'шт'),    # что → што
    (r'гк', 'хк'),    # мягкий → мяхкий
    (r'гч', 'хч'),
    (r'сч', 'щ'),     # счастье → щастье
    (r'зч', 'щ'),
    (r'сш', 'ш'),     # сшить → шить (долгий ш)
    (r'зж', 'ж'),     # разжечь → ражжечь
    (r'стн', 'сн'),   # честный → чесный
    (r'здн', 'зн'),   # поздно → позно
    (r'стл', 'сл'),   # счастливый → щасливый
    (r'рдц', 'рц'),   # сердце → серце
    (r'лнц', 'нц'),   # солнце → сонце
    (r'вств', 'ств'), # чувство → чуство
    (r'ндск', 'нск'), # голландский → голланский
    (r'нтск', 'нск'),
]


@lru_cache(maxsize=10000)
def phonetic_normalize(word: str) -> str:
    """
    Фонетическая нормализация слова.

    Приводит слово к фонетическому виду,
    убирая непроизносимые согласные и редуцируя гласные.
    """
    result = normalize_word(word)

    for pattern, replacement in PHONETIC_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)

    return result


# =============================================================================
# БЕЗОПАСНЫЕ ПАДЕЖНЫЕ ПАРЫ
# =============================================================================

# Падежи, которые часто звучат одинаково для определённых типов слов
SAFE_CASE_PAIRS: Dict[frozenset, Set[str]] = {
    # Именительный и Винительный — для неодушевлённых существительных
    frozenset({'nomn', 'accs'}): {'NOUN', 'ADJF'},

    # Родительный и Винительный — для одушевлённых мужского рода
    frozenset({'gent', 'accs'}): {'NOUN'},  # только для anim

    # Дательный и Предложный — некоторые окончания совпадают
    frozenset({'datv', 'loct'}): set(),  # требует доп. проверки окончаний
}


# =============================================================================
# КЛАСС УМНЫХ ПРАВИЛ
# =============================================================================

@dataclass
class RuleResult:
    """Результат применения правила."""
    is_match: bool          # True = фильтровать, False = реальная ошибка (не фильтровать!)
    rule_name: str
    confidence: float       # 0.0 - 1.0
    details: str = ""
    is_real_error: bool = False  # v2.0: явный сигнал "это реальная ошибка"


class SmartRules:
    """
    Умные правила фильтрации на основе морфологии.

    Заменяют статические списки пар на алгоритмическую проверку.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация.

        Args:
            config: Конфигурация (пороги, включение/выключение правил)
        """
        self.config = config or {}

        # Пороги по умолчанию
        self.levenshtein_threshold = self.config.get('levenshtein_threshold', 2)
        # v1.1: Повышен порог phonetic с 85% до 95% для уменьшения ложных фильтраций
        self.phonetic_threshold = self.config.get('phonetic_threshold', 95)  # %
        self.min_word_len = self.config.get('min_word_len', 3)

        # Включённые правила
        self.enabled_rules = self.config.get('enabled_rules', {
            # v2.0: Новые правила заменяют старые
            'lemma': True,       # Заменяет same_lemma
            'homophone': True,   # Заменяет homophone
            'levenshtein': True, # Заменяет levenshtein_similar_lemma
            'grammar': True,     # Заменяет grammar_ending
            # Оригинальные правила
            'aspect_pair': True,
            'safe_case': True,
            'phonetic': True,
            'numeral': True,
            'participle': True,
            'reflexive': True,
            'prefix': True,
        })

    def is_false_positive(self, word1: str, word2: str) -> Optional[RuleResult]:
        """
        Проверяет, является ли пара ложным срабатыванием.

        Returns:
            RuleResult если это ложное срабатывание, иначе None
        """
        w1 = normalize_word(word1)
        w2 = normalize_word(word2)

        if w1 == w2:
            return RuleResult(True, 'identical', 1.0, 'Слова идентичны')

        # Слишком короткие слова — отдельная логика
        if len(w1) < self.min_word_len or len(w2) < self.min_word_len:
            return None  # Короткие слова требуют отдельной проверки

        # Проверяем правила по порядку
        # v2.0: Новые правила ПЕРВЫМИ — они заменяют старые
        rules = [
            # Замена старых правил (порядок важен!)
            ('lemma', self._check_lemma),           # Заменяет same_lemma
            ('homophone', self._check_homophone),   # Заменяет homophone
            ('levenshtein', self._check_levenshtein), # Заменяет levenshtein_similar_lemma
            ('grammar', self._check_grammar),       # Заменяет grammar_ending
            # Оригинальные правила
            ('aspect_pair', self._check_aspect_pair),
            ('safe_case', self._check_safe_case),
            ('reflexive', self._check_reflexive),
            ('participle', self._check_participle),
            ('numeral', self._check_numeral),
            ('prefix', self._check_prefix),
            ('phonetic', self._check_phonetic),
        ]

        for rule_name, rule_func in rules:
            if not self.enabled_rules.get(rule_name, True):
                continue

            result = rule_func(w1, w2)
            if result and result.is_match:
                return result

        return None

    # =========================================================================
    # v2.0: НОВЫЕ ПРАВИЛА, ЗАМЕНЯЮЩИЕ СТАРЫЕ
    # =========================================================================

    def _check_lemma(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка одинаковой леммы (заменяет same_lemma).

        Фильтрует слова с одной леммой, НО проверяет опасные случаи:
        - Разное число (будут/будет) — реальная ошибка
        - Разный падеж (преграды/преград) — реальная ошибка
        - Разное время глаголов (получилось/получится) — реальная ошибка
        - Приставка "по-" (больше/побольше) — реальная ошибка
        - VERB↔GRND (делая/делать) — реальная ошибка
        - VERB↔PRTF (причастие vs глагол) — реальная ошибка
        """
        if not HAS_MORPHOLOGY:
            return None

        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
        if lemma1 != lemma2:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)

        # Деепричастия — не фильтруем переход в другую форму
        if pos1 == 'GRND' or pos2 == 'GRND':
            if pos1 != pos2:
                return None

        # Причастия — не фильтруем переход в другую форму
        participles = {'PRTF', 'PRTS'}
        if (pos1 in participles) != (pos2 in participles):
            return None

        # Разное число — реальная ошибка
        num1, num2 = get_number(w1), get_number(w2)
        if num1 and num2 and num1 != num2:
            return None

        # ПРИЛАГАТЕЛЬНЫЕ: проверяем падеж
        adjectives = {'ADJF', 'ADJS'}
        if pos1 in adjectives and pos2 in adjectives:
            case1, case2 = get_case(w1), get_case(w2)
            if case1 and case2 and case1 != case2:
                # Безопасные пары падежей (звучат одинаково)
                safe_pairs = {frozenset({'nomn', 'accs'})}
                if frozenset({case1, case2}) not in safe_pairs:
                    return None

        # ГЛАГОЛЫ: проверяем время
        verbs = {'VERB', 'INFN'}
        if pos1 in verbs and pos2 in verbs:
            tense1, tense2 = get_tense(w1), get_tense(w2)
            if tense1 and tense2 and tense1 != tense2:
                return None

        # СУЩЕСТВИТЕЛЬНЫЕ: проверяем падеж
        if pos1 == 'NOUN' and pos2 == 'NOUN':
            case1, case2 = get_case(w1), get_case(w2)
            if case1 and case2 and case1 != case2:
                # Безопасные пары падежей
                safe_pairs = {frozenset({'nomn', 'accs'})}
                if frozenset({case1, case2}) not in safe_pairs:
                    return None

        # Приставка "по-" — реальная ошибка (больше/побольше)
        if w1.startswith('по') and len(w1) > 3 and w1[2:] == w2:
            return None
        if w2.startswith('по') and len(w2) > 3 and w2[2:] == w1:
            return None

        return RuleResult(
            True, 'lemma', 0.95,
            f'Одинаковая лемма: {lemma1}'
        )

    def _check_homophone(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка омофонов (заменяет homophone).

        Использует фонетическую нормализацию вместо статического словаря.
        Слова-омофоны звучат одинаково, но пишутся по-разному.

        ВАЖНО: Не фильтруем, если это одна лемма с разным числом/падежом —
        это реальная ошибка чтеца, а не омофон!
        """
        # Сначала проверяем морфологию
        if HAS_MORPHOLOGY:
            lemma1, lemma2 = get_lemma(w1), get_lemma(w2)

            # Если одна лемма — проверяем опасные различия
            if lemma1 == lemma2:
                # Разное число — реальная ошибка (сотни/сотня)
                num1, num2 = get_number(w1), get_number(w2)
                if num1 and num2 and num1 != num2:
                    return None

                # Разный падеж — реальная ошибка
                case1, case2 = get_case(w1), get_case(w2)
                if case1 and case2 and case1 != case2:
                    # Безопасные пары
                    safe_pairs = {frozenset({'nomn', 'accs'})}
                    if frozenset({case1, case2}) not in safe_pairs:
                        return None

        # Фонетическая нормализация
        pn1 = phonetic_normalize(w1)
        pn2 = phonetic_normalize(w2)

        if pn1 == pn2:
            return RuleResult(
                True, 'homophone', 0.95,
                f'Омофоны: [{pn1}]'
            )

        return None

    def _check_levenshtein(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка морфологически близких слов (заменяет levenshtein_similar_lemma).

        Фильтрует слова с близкой леммой по Левенштейну, НО проверяет:
        - Разная часть речи — реальная ошибка
        - Разные глаголы (видовые пары) — реальная ошибка
        - Разные местоимения — реальная ошибка
        - Разные наречия — реальная ошибка
        - Разные прилагательные — реальная ошибка
        """
        if not HAS_MORPHOLOGY:
            return None

        # Проверяем расстояние между словами
        dist = levenshtein_distance(w1, w2)
        if dist > self.levenshtein_threshold:
            return None

        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)

        # Одинаковая лемма — уже обработано в _check_lemma
        if lemma1 == lemma2:
            return None

        # Проверяем расстояние между леммами
        lemma_dist = levenshtein_distance(lemma1, lemma2)
        if lemma_dist > 1:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)

        # Разная часть речи — не фильтруем
        if pos1 and pos2 and pos1 != pos2:
            return None

        # Разные глаголы (видовые пары обрабатываются отдельно)
        if pos1 in ('VERB', 'INFN') and pos2 in ('VERB', 'INFN'):
            return None

        # Разные местоимения
        if pos1 == 'NPRO' and pos2 == 'NPRO':
            return None

        # Разные наречия
        if pos1 == 'ADVB' and pos2 == 'ADVB':
            return None

        # Разные прилагательные
        if pos1 in ('ADJF', 'ADJS') and pos2 in ('ADJF', 'ADJS'):
            return None

        return RuleResult(
            True, 'levenshtein', 0.85,
            f'Близкие леммы: {lemma1} / {lemma2}'
        )

    def _check_grammar(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка грамматических вариаций (заменяет grammar_ending).

        Фильтрует слова с одной леммой и разными грамматическими окончаниями,
        НО проверяет опасные случаи (см. _check_lemma).
        """
        if not HAS_MORPHOLOGY:
            return None

        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
        if lemma1 != lemma2:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)

        # VERB↔GRND — реальная ошибка
        if (pos1 == 'VERB' and pos2 == 'GRND') or (pos1 == 'GRND' and pos2 == 'VERB'):
            return None

        # Разное число — реальная ошибка
        num1, num2 = get_number(w1), get_number(w2)
        if num1 and num2 and num1 != num2:
            return None

        # Типичные грамматические окончания
        grammar_endings = [
            ('ого', 'его'),
            ('ому', 'ему'),
            ('ой', 'ей'),
            ('ые', 'ие'),
            ('ым', 'им'),
            ('ых', 'их'),
            ('ая', 'яя'),
            ('ое', 'ее'),
            ('ую', 'юю'),
        ]

        for end1, end2 in grammar_endings:
            if (w1.endswith(end1) and w2.endswith(end2)) or \
               (w1.endswith(end2) and w2.endswith(end1)):
                return RuleResult(
                    True, 'grammar', 0.9,
                    f'Грамматическое окончание: -{end1}/-{end2}'
                )

        return None

    # =========================================================================
    # ОРИГИНАЛЬНЫЕ ПРАВИЛА
    # =========================================================================

    def _check_aspect_pair(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка видовых пар глаголов.

        v1.1: Убрана фильтрация "супплетивных пар" — только реальные видовые пары
        с одной основой (делать/сделать), но не разные глаголы (ощущать/ощутить)
        """
        if not HAS_MORPHOLOGY:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)
        verb_pos = {'VERB', 'INFN'}

        # Оба должны быть глаголами
        if pos1 not in verb_pos or pos2 not in verb_pos:
            return None

        # Проверяем видовую пару И общую основу
        # v1.1: Требуем ОБЯЗАТЕЛЬНО одну основу, иначе это разные глаголы
        if is_aspect_pair(w1, w2) and is_same_verb_base(w1, w2):
            return RuleResult(
                True, 'aspect_pair', 0.95,
                f'Видовая пара: {get_aspect(w1)} / {get_aspect(w2)}'
            )

        return None

    def _check_safe_case(self, w1: str, w2: str) -> Optional[RuleResult]:
        """Проверка безопасных падежных вариаций."""
        if not HAS_MORPHOLOGY:
            return None

        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
        if lemma1 != lemma2:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)
        if pos1 != pos2:
            return None

        case1, case2 = get_case(w1), get_case(w2)
        if not case1 or not case2 or case1 == case2:
            return None

        num1, num2 = get_number(w1), get_number(w2)
        if num1 and num2 and num1 != num2:
            return None  # Разное число — реальная ошибка

        # Проверяем безопасные пары
        case_pair = frozenset({case1, case2})

        # Именительный/Винительный
        if case_pair == frozenset({'nomn', 'accs'}):
            if pos1 in {'NOUN', 'ADJF'}:
                # Для неодушевлённых — безопасно
                # Для одушевлённых м.р. — вин. ≠ им., но = род.
                # Определяем одушевлённость через pymorphy
                if HAS_PYMORPHY and morph:
                    p1 = morph.parse(w1)[0]
                    if 'anim' in p1.tag:
                        return None  # Одушевлённое — не фильтруем

                return RuleResult(
                    True, 'safe_case', 0.9,
                    f'Им./Вин. для неодушевлённого: {case1} → {case2}'
                )

        return None

    def _check_reflexive(self, w1: str, w2: str) -> Optional[RuleResult]:
        """Проверка возвратных форм (-ся/-сь)."""
        # Проверяем, отличаются ли только -ся/-сь
        for suf1, suf2 in [('ся', 'сь'), ('сь', 'ся')]:
            if w1.endswith(suf1) and w2.endswith(suf2):
                base1 = w1[:-len(suf1)]
                base2 = w2[:-len(suf2)]
                if base1 == base2:
                    return RuleResult(
                        True, 'reflexive', 0.95,
                        f'Возвратные формы: -{suf1} / -{suf2}'
                    )

        # Рефлексивный vs нерефлексивный глагол с одной основой
        if (w1.endswith('ся') or w1.endswith('сь')) != (w2.endswith('ся') or w2.endswith('сь')):
            # Убираем суффикс и сравниваем
            base1 = re.sub(r'(ся|сь)$', '', w1)
            base2 = re.sub(r'(ся|сь)$', '', w2)
            if base1 == base2 or levenshtein_distance(base1, base2) <= 1:
                if HAS_MORPHOLOGY:
                    lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
                    # Проверяем, что это формы одного глагола
                    if lemma1.rstrip('ся').rstrip('сь') == lemma2.rstrip('ся').rstrip('сь'):
                        return RuleResult(
                            True, 'reflexive', 0.85,
                            f'Возвратная/невозвратная форма одного глагола'
                        )

        return None

    def _check_participle(self, w1: str, w2: str) -> Optional[RuleResult]:
        """Проверка причастий и их форм."""
        if not HAS_MORPHOLOGY:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)
        participle_pos = {'PRTF', 'PRTS'}  # полное и краткое причастие

        # Одно причастие?
        if pos1 in participle_pos or pos2 in participle_pos:
            lemma1, lemma2 = get_lemma(w1), get_lemma(w2)

            if lemma1 == lemma2:
                # Полное vs краткое причастие
                if (pos1 in participle_pos) and (pos2 in participle_pos):
                    return RuleResult(
                        True, 'participle', 0.9,
                        f'Формы причастия: {pos1} / {pos2}'
                    )

                # Причастие vs глагол (прич. образовано от глагола)
                verb_pos = {'VERB', 'INFN'}
                if (pos1 in participle_pos and pos2 in verb_pos) or \
                   (pos2 in participle_pos and pos1 in verb_pos):
                    # Проверяем, что глагольная основа совпадает
                    return RuleResult(
                        True, 'participle', 0.8,
                        f'Причастие/глагол одной основы'
                    )

        return None

    def _check_numeral(self, w1: str, w2: str) -> Optional[RuleResult]:
        """Проверка числительных."""
        if not HAS_MORPHOLOGY:
            return None

        pos1, pos2 = get_pos(w1), get_pos(w2)
        numeral_pos = {'NUMR', 'NUMB'}  # числительные

        if pos1 in numeral_pos or pos2 in numeral_pos:
            lemma1, lemma2 = get_lemma(w1), get_lemma(w2)

            if lemma1 == lemma2:
                return RuleResult(
                    True, 'numeral', 0.9,
                    f'Формы числительного'
                )

            # Проверяем схожие числительные (одиннадцать/одинадцать)
            if levenshtein_distance(lemma1, lemma2) <= 2:
                return RuleResult(
                    True, 'numeral', 0.7,
                    f'Схожие числительные'
                )

        return None

    def _check_prefix(self, w1: str, w2: str) -> Optional[RuleResult]:
        """Проверка приставочных вариантов."""
        if not HAS_MORPHOLOGY:
            return None

        # Типичные приставки
        prefixes = ['не', 'на', 'по', 'от', 'у', 'вы', 'за', 'до', 'под', 'при', 'пере', 'с', 'в', 'об', 'про']

        short, long = (w1, w2) if len(w1) < len(w2) else (w2, w1)

        for prefix in prefixes:
            if long.startswith(prefix):
                base = long[len(prefix):]
                if base == short:
                    # Проверяем, что base — валидное слово
                    if HAS_PYMORPHY and morph:
                        parsed = morph.parse(short)
                        if parsed and parsed[0].score > 0.1:
                            pos = parsed[0].tag.POS
                            # Только для глаголов — это может быть видовая пара
                            if pos in {'VERB', 'INFN'}:
                                return RuleResult(
                                    True, 'prefix', 0.85,
                                    f'Приставочный вариант глагола: {prefix}+'
                                )

        return None

    def _check_phonetic(self, w1: str, w2: str) -> Optional[RuleResult]:
        """
        Проверка фонетического сходства.

        v1.1: Добавлена проверка леммы — если леммы разные, фонетическое
        сходство не является основанием для фильтрации (это могут быть
        разные слова: старейший/старший, проводит/проходит)

        v1.1.1: Добавлена проверка падежа — если одна лемма, но разный падеж,
        это реальная ошибка (соглядатаи/соглядатаев — nomn/gent)
        """
        # v1.1: Сначала проверяем лемму
        if HAS_MORPHOLOGY:
            lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
            if lemma1 != lemma2:
                # Разные леммы — фонетическое сходство не релевантно
                return None

            # v1.1.1: Если одна лемма, но разный падеж — это реальная ошибка
            case1, case2 = get_case(w1), get_case(w2)
            if case1 and case2 and case1 != case2:
                # Падежная замена — реальная ошибка, не фильтруем
                return None

            # v1.1.1: Если одна лемма, но разное число — это реальная ошибка
            num1, num2 = get_number(w1), get_number(w2)
            if num1 and num2 and num1 != num2:
                # Замена числа — реальная ошибка, не фильтруем
                return None

        # Фонетическая нормализация
        pn1 = phonetic_normalize(w1)
        pn2 = phonetic_normalize(w2)

        if pn1 == pn2:
            return RuleResult(
                True, 'phonetic', 0.9,
                f'Фонетически идентичны: [{pn1}]'
            )

        # Проверяем схожесть фонетических форм
        ratio = levenshtein_ratio(pn1, pn2)
        if ratio >= self.phonetic_threshold:
            return RuleResult(
                True, 'phonetic', ratio / 100,
                f'Фонетически схожи ({ratio}%): [{pn1}] / [{pn2}]'
            )

        return None

    def get_all_results(self, word1: str, word2: str) -> List[RuleResult]:
        """
        Возвращает результаты всех правил (для отладки).
        """
        w1 = normalize_word(word1)
        w2 = normalize_word(word2)

        results = []

        rules = [
            # v2.0: Новые правила
            ('lemma', self._check_lemma),
            ('homophone', self._check_homophone),
            ('levenshtein', self._check_levenshtein),
            ('grammar', self._check_grammar),
            # Оригинальные правила
            ('aspect_pair', self._check_aspect_pair),
            ('safe_case', self._check_safe_case),
            ('reflexive', self._check_reflexive),
            ('participle', self._check_participle),
            ('numeral', self._check_numeral),
            ('prefix', self._check_prefix),
            ('phonetic', self._check_phonetic),
        ]

        for rule_name, rule_func in rules:
            try:
                result = rule_func(w1, w2)
                if result:
                    results.append(result)
            except Exception as e:
                results.append(RuleResult(False, rule_name, 0.0, f'Error: {e}'))

        return results


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_smart_rules_instance: Optional[SmartRules] = None


def get_smart_rules() -> SmartRules:
    """Возвращает глобальный экземпляр SmartRules."""
    global _smart_rules_instance
    if _smart_rules_instance is None:
        _smart_rules_instance = SmartRules()
    return _smart_rules_instance


# =============================================================================
# УДОБНЫЕ ФУНКЦИИ
# =============================================================================

def is_smart_false_positive(word1: str, word2: str) -> bool:
    """
    Быстрая проверка ложного срабатывания.

    Returns:
        True если это ложное срабатывание по умным правилам
    """
    result = get_smart_rules().is_false_positive(word1, word2)
    return result is not None and result.is_match


def get_false_positive_reason(word1: str, word2: str) -> Optional[str]:
    """
    Возвращает причину ложного срабатывания.

    Returns:
        Описание правила или None
    """
    result = get_smart_rules().is_false_positive(word1, word2)
    if result and result.is_match:
        return f"{result.rule_name}: {result.details}"
    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI для тестирования."""
    import argparse

    parser = argparse.ArgumentParser(description='Smart Rules — умные правила фильтрации')
    parser.add_argument('word1', nargs='?', help='Первое слово')
    parser.add_argument('word2', nargs='?', help='Второе слово')
    parser.add_argument('--all', '-a', action='store_true', help='Показать все правила')
    parser.add_argument('--version', '-V', action='store_true', help='Версия')

    args = parser.parse_args()

    if args.version:
        print(f"Smart Rules v{VERSION} ({VERSION_DATE})")
        print(f"  Morphology: {'доступна' if HAS_MORPHOLOGY else 'недоступна'}")
        return

    if not args.word1 or not args.word2:
        # Демо-тест
        pairs = [
            ('делать', 'сделать'),
            ('писать', 'написать'),
            ('встречаться', 'встречаться'),
            ('получился', 'получилось'),
            ('читая', 'читаясь'),
            ('дом', 'дома'),
            ('солнце', 'сонце'),
            ('счастье', 'щастье'),
            ('что', 'што'),
        ]

        print("\n=== Демо Smart Rules ===\n")
        rules = get_smart_rules()

        for w1, w2 in pairs:
            result = rules.is_false_positive(w1, w2)
            status = "✓ FP" if result and result.is_match else "✗"
            details = f" ({result.rule_name}: {result.confidence:.0%})" if result else ""
            print(f"  {w1} / {w2}: {status}{details}")

        return

    rules = get_smart_rules()

    if args.all:
        results = rules.get_all_results(args.word1, args.word2)
        print(f"\n=== Все правила для '{args.word1}' / '{args.word2}' ===\n")
        for r in results:
            status = "✓" if r.is_match else "✗"
            print(f"  {status} {r.rule_name}: {r.confidence:.0%} — {r.details}")
        if not results:
            print("  (нет совпадений)")
    else:
        result = rules.is_false_positive(args.word1, args.word2)
        if result and result.is_match:
            print(f"✓ Ложное срабатывание: {result.rule_name}")
            print(f"  Уверенность: {result.confidence:.0%}")
            print(f"  Детали: {result.details}")
        else:
            print(f"✗ Не ложное срабатывание")


if __name__ == '__main__':
    main()
