"""
Learned Rules v1.0 — Правила фильтрации, обученные на данных.

Обучено на 614 парах:
- 70 реальных ошибок (golden tests)
- 544 ложных ошибок (отфильтрованных v5.7)

Ключевые инсайты из данных:
- any_name: 100% FP — имена собственные ВСЕГДА фильтруем
- same_phonetic + same_num + same_pos: 98% FP — омофоны
- same_lemma: 89% FP, НО verb_grnd исключаем (74% FP — много реальных)
- different_num: указывает на реальную ошибку

v1.0 (2026-01-26): Начальная версия
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-26'

from typing import Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from morphology import (
        normalize_word, get_lemma, get_pos, get_number, get_case,
        HAS_PYMORPHY, morph
    )
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False
    HAS_PYMORPHY = False
    morph = None

try:
    from .comparison import levenshtein_distance
    from .smart_rules import phonetic_normalize
except ImportError:
    from comparison import levenshtein_distance
    from smart_rules import phonetic_normalize


@dataclass
class LearnedResult:
    """Результат применения обученного правила."""
    is_false_positive: bool
    rule_name: str
    confidence: float
    details: str = ""


class LearnedRules:
    """
    Правила фильтрации, обученные на данных.

    Приоритет правил (от высшего к низшему):
    1. verb_grnd → НЕ фильтровать (74% — много реальных ошибок)
    2. different_num → НЕ фильтровать (разное число = реальная ошибка)
    3. any_name → фильтровать (100% FP)
    4. same_phonetic + same_pos → фильтровать (98% FP)
    5. same_lemma → фильтровать (89% FP)
    """

    def __init__(self):
        pass

    def is_false_positive(self, w1: str, w2: str) -> Optional[LearnedResult]:
        """
        Определяет, является ли пара ложной ошибкой.

        Returns:
            LearnedResult если можно определить, иначе None
        """
        w1 = normalize_word(w1)
        w2 = normalize_word(w2)

        if w1 == w2:
            return LearnedResult(True, 'identical', 1.0, 'Слова идентичны')

        if not HAS_MORPHOLOGY:
            return None

        # Извлекаем признаки
        lemma1, lemma2 = get_lemma(w1), get_lemma(w2)
        pos1, pos2 = get_pos(w1), get_pos(w2)
        num1, num2 = get_number(w1), get_number(w2)
        case1, case2 = get_case(w1), get_case(w2)

        same_lemma = lemma1 == lemma2
        same_pos = pos1 == pos2
        same_num = num1 == num2 if num1 and num2 else None
        same_phonetic = phonetic_normalize(w1) == phonetic_normalize(w2)

        # VERB↔GRND проверка
        verb_grnd = (pos1 == 'VERB' and pos2 == 'GRND') or (pos1 == 'GRND' and pos2 == 'VERB')

        # Имя собственное
        p1 = morph.parse(w1)[0] if morph else None
        p2 = morph.parse(w2)[0] if morph else None
        any_name = False
        if p1 and p2:
            any_name = 'Name' in str(p1.tag) or 'Name' in str(p2.tag)

        # ========================================
        # ПРАВИЛО 1: VERB↔GRND → НЕ фильтровать
        # 74% FP — много реальных ошибок
        # ========================================
        if verb_grnd:
            return LearnedResult(False, 'verb_grnd_protected', 0.74,
                                'VERB↔GRND — часто реальная ошибка')

        # ========================================
        # ПРАВИЛО 2: Разное число → НЕ фильтровать
        # Данные показывают: разное число = реальная ошибка
        # ========================================
        if same_lemma and same_num == False:
            return LearnedResult(False, 'different_num_protected', 0.85,
                                'Одна лемма, разное число — реальная ошибка')

        # ========================================
        # ПРАВИЛО 3: Имя собственное → фильтровать
        # 100% FP в данных
        # ========================================
        if any_name:
            return LearnedResult(True, 'name', 1.0,
                                'Имя собственное — ошибка транскрипции')

        # ========================================
        # ПРАВИЛО 4: Омофоны → фильтровать
        # 98% FP (если same_pos)
        # ========================================
        if same_phonetic and same_pos:
            return LearnedResult(True, 'phonetic', 0.98,
                                f'Фонетически идентичны: [{phonetic_normalize(w1)}]')

        # ========================================
        # ПРАВИЛО 5: Одинаковая лемма → фильтровать
        # 89% FP (verb_grnd и different_num уже исключены)
        # ========================================
        if same_lemma:
            return LearnedResult(True, 'lemma', 0.89,
                                f'Одинаковая лемма: {lemma1}')

        # ========================================
        # ПРАВИЛО 6: Близкие леммы (levenshtein ≤ 1) → фильтровать
        # Но не для разных POS
        # ========================================
        if lemma1 and lemma2:
            lemma_dist = levenshtein_distance(lemma1, lemma2)
            if lemma_dist <= 1 and same_pos:
                # Исключения: местоимения, наречия, прилагательные с разной леммой
                dangerous_pos = {'NPRO', 'ADVB', 'ADJF', 'ADJS', 'VERB', 'INFN'}
                if pos1 not in dangerous_pos:
                    return LearnedResult(True, 'similar_lemma', 0.85,
                                        f'Близкие леммы: {lemma1}/{lemma2}')

        return None


# Singleton
_learned_rules_instance = None

def get_learned_rules() -> LearnedRules:
    """Возвращает глобальный экземпляр LearnedRules."""
    global _learned_rules_instance
    if _learned_rules_instance is None:
        _learned_rules_instance = LearnedRules()
    return _learned_rules_instance


def is_learned_false_positive(w1: str, w2: str) -> Tuple[bool, str]:
    """
    Быстрая проверка через обученные правила.

    Returns:
        (is_fp, reason) — является ли ложной ошибкой и причина
    """
    result = get_learned_rules().is_false_positive(w1, w2)
    if result:
        return result.is_false_positive, result.rule_name
    return False, 'unknown'
