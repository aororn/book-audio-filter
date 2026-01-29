"""
Функции сравнения слов для фильтрации ошибок транскрипции.

Содержит:
- normalize_word — импорт из morphology.py (единый источник)
- levenshtein_distance / levenshtein_ratio — расстояние Левенштейна
- is_homophone_match — проверка омофонов
- is_grammar_ending_match — проверка грамматических окончаний
- is_case_form_match — проверка падежных форм
- is_adverb_adjective_match — наречие↔прилагательное
- is_verb_gerund_safe_match — глагол↔деепричастие (безопасные)
- is_short_full_adjective_match — краткое↔полное прилагательное
- is_lemma_match — формы одного слова
- is_similar_by_levenshtein — схожесть по Левенштейну
- is_yandex_typical_error — типичная ошибка Яндекса
- is_prefix_variant — варианты с приставками
- is_aspect_pair — видовые пары глаголов (v6.0)
- is_safe_case_variation — расширенные падежные пары (v6.0)
- phonetic_normalize — фонетическая нормализация (v6.0)

v5.7.0: Унификация — normalize_word импортируется из morphology.py
v6.0.0: Убрано дублирование pymorphy, добавлены aspect/phonetic функции
"""

from functools import lru_cache
from typing import Optional, Tuple

from .constants import (
    HOMOPHONES, HOMOPHONE_PATTERNS_COMPILED, GRAMMAR_ENDINGS,
    YANDEX_TYPICAL_ERRORS, YANDEX_PREFIX_ERRORS,
)

# =============================================================================
# ИМПОРТ ЗАВИСИМОСТЕЙ
# =============================================================================

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    try:
        import Levenshtein as lev
        HAS_RAPIDFUZZ = False
        HAS_LEVENSHTEIN = True
    except ImportError:
        HAS_RAPIDFUZZ = False
        HAS_LEVENSHTEIN = False

# =============================================================================
# ИМПОРТ ИЗ MORPHOLOGY.PY — ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ
# v6.0: Убрано локальное создание pymorphy2.MorphAnalyzer() — используем из morphology
# =============================================================================

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта morphology
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from morphology import (
        normalize_word,
        parse_word_cached,
        get_word_info,
        get_lemma,
        get_pos,
        get_number,
        get_gender,
        get_case,
        HAS_PYMORPHY,
        morph,  # v6.0: Используем единый экземпляр MorphAnalyzer
    )
except ImportError as e:
    # Fallback — если morphology.py недоступен
    import warnings
    warnings.warn(f"morphology.py недоступен: {e}. Используем fallback.")

    HAS_PYMORPHY = False
    morph = None

    try:
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        HAS_PYMORPHY = True
    except ImportError:
        pass

    @lru_cache(maxsize=10000)
    def parse_word_cached(word: str) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Кэшированный разбор слова через pymorphy (fallback)."""
        if not HAS_PYMORPHY or not morph:
            return (word, None, None, None, None)
        parsed = morph.parse(word)
        if not parsed:
            return (word, None, None, None, None)
        p = parsed[0]
        tag = p.tag
        lemma = p.normal_form
        pos = tag.POS
        number = 'sing' if 'sing' in tag else ('plur' if 'plur' in tag else None)
        gender = 'masc' if 'masc' in tag else ('femn' if 'femn' in tag else ('neut' if 'neut' in tag else None))
        case = None
        for c in ('nomn', 'gent', 'datv', 'accs', 'ablt', 'loct'):
            if c in tag:
                case = c
                break
        return (lemma, pos, number, gender, case)

    def normalize_word(word: str) -> str:
        return word.lower().strip().replace('ё', 'е')

    def get_word_info(word: str) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
        return parse_word_cached(normalize_word(word))

    def get_lemma(word: str) -> str:
        return get_word_info(word)[0]

    def get_pos(word: str) -> Optional[str]:
        return get_word_info(word)[1]

    def get_number(word: str) -> Optional[str]:
        return get_word_info(word)[2]

    def get_gender(word: str) -> Optional[str]:
        return get_word_info(word)[3]

    def get_case(word: str) -> Optional[str]:
        return get_word_info(word)[4]


# =============================================================================
# РАССТОЯНИЕ ЛЕВЕНШТЕЙНА
# =============================================================================

@lru_cache(maxsize=50000)
def levenshtein_distance(s1: str, s2: str) -> int:
    """Кэшированное расстояние Левенштейна."""
    if HAS_RAPIDFUZZ:
        return Levenshtein.distance(s1, s2)
    elif not HAS_RAPIDFUZZ and 'HAS_LEVENSHTEIN' in dir() and HAS_LEVENSHTEIN:
        return lev.distance(s1, s2)
    else:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


def levenshtein_ratio(s1: str, s2: str) -> int:
    """Коэффициент схожести (0-100)."""
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2)
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return int((1 - dist / max_len) * 100) if max_len else 100


# =============================================================================
# ФУНКЦИИ СРАВНЕНИЯ
# =============================================================================

def is_homophone_match(word1: str, word2: str) -> bool:
    """Проверяет омофоны."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if w1 == w2:
        return True

    if (w1, w2) in HOMOPHONES or (w2, w1) in HOMOPHONES:
        return True

    for pattern, replacement in HOMOPHONE_PATTERNS_COMPILED:
        w1_alt = pattern.sub(replacement, w1)
        w2_alt = pattern.sub(replacement, w2)
        if w1_alt == w2 or w2_alt == w1 or w1_alt == w2_alt:
            return True

    if len(w1) > 3 and len(w2) > 3 and w1[:-2] == w2[:-2]:
        end1, end2 = w1[-2:], w2[-2:]
        if (end1, end2) in HOMOPHONES or (end2, end1) in HOMOPHONES:
            return True

    return False


def is_grammar_ending_match(word1: str, word2: str) -> bool:
    """Проверяет грамматические окончания."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if len(w1) < 3 or len(w2) < 3:
        return False

    min_base = 3
    max_end = 4

    for base_len in range(min(len(w1), len(w2)) - 1, min_base - 1, -1):
        if w1[:base_len] == w2[:base_len]:
            end1 = w1[base_len:]
            end2 = w2[base_len:]
            if len(end1) <= max_end and len(end2) <= max_end:
                if (end1, end2) in GRAMMAR_ENDINGS or (end2, end1) in GRAMMAR_ENDINGS:
                    return True
            break

    return False


def is_case_form_match(word1: str, word2: str) -> bool:
    """Проверяет падежные формы одного слова.

    ВАЖНО: Фильтруем только незначимые падежные различия (им. vs вин. для неодушевлённых).
    Различия в числе или значимые падежные различия (род. vs им.) — это реальные ошибки!
    """
    if not HAS_PYMORPHY:
        return False

    info1 = get_word_info(word1)
    info2 = get_word_info(word2)
    lemma1, pos1, num1, _, case1 = info1
    lemma2, pos2, num2, _, case2 = info2

    if lemma1 != lemma2:
        return False
    if pos1 != pos2:
        return False
    if pos1 != 'NOUN':
        return False

    # ВАЖНО: Если разное число — это реальная ошибка, не фильтруем!
    # Например: "десятки" vs "десятках" — разное число, это ошибка чтеца
    if num1 and num2 and num1 != num2:
        return False

    # Фильтруем только незначимые падежные различия:
    # - им. vs вин. для неодушевлённых (одинаково звучат)
    # Но НЕ фильтруем род. vs им./вин. — это разные конструкции!
    if case1 and case2 and case1 != case2:
        # Безопасные пары падежей для фильтрации (звучат одинаково)
        safe_pairs = {
            frozenset({'nomn', 'accs'}),  # им. и вин. для неодушевлённых
        }
        if frozenset({case1, case2}) in safe_pairs:
            return True
        # Все остальные падежные различия — реальные ошибки, не фильтруем
        return False

    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    # Специфические паттерны окончаний (оставляем для совместимости)
    if w1.endswith('ей') and w2.endswith('и') and w1[:-2] == w2[:-1]:
        return True
    if w2.endswith('ей') and w1.endswith('и') and w2[:-2] == w1[:-1]:
        return True
    if w1.endswith('ей') and w2.endswith('еи') and w1[:-1] == w2[:-1]:
        return True
    if w2.endswith('ей') and w1.endswith('еи') and w2[:-1] == w1[:-1]:
        return True

    return False


def is_adverb_adjective_match(word1: str, word2: str) -> bool:
    """Проверяет пары наречие↔прилагательное."""
    if not HAS_PYMORPHY:
        return False

    info1 = get_word_info(word1)
    info2 = get_word_info(word2)
    lemma1, pos1, _, _, _ = info1
    lemma2, pos2, _, _, _ = info2

    if lemma1 != lemma2 and levenshtein_distance(lemma1, lemma2) > 2:
        return False

    adv_adj_pairs = {
        ('ADVB', 'ADJF'), ('ADVB', 'ADJS'),
    }

    pair = (pos1, pos2) if pos1 else (None, None)
    pair_rev = (pos2, pos1)

    if pair in adv_adj_pairs or pair_rev in adv_adj_pairs:
        return True

    return False


def is_verb_gerund_safe_match(word1: str, word2: str) -> bool:
    """Проверяет безопасные пары глагол↔деепричастие."""
    if not HAS_PYMORPHY:
        return False

    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    info1 = get_word_info(w1)
    info2 = get_word_info(w2)
    lemma1, pos1, _, _, _ = info1
    lemma2, pos2, _, _, _ = info2

    if lemma1 != lemma2:
        return False

    verb_pos = {'VERB', 'INFN'}
    gerund_pos = {'GRND'}

    is_verb1 = pos1 in verb_pos
    is_verb2 = pos2 in verb_pos
    is_gerund1 = pos1 in gerund_pos
    is_gerund2 = pos2 in gerund_pos

    if not ((is_verb1 and is_gerund2) or (is_gerund1 and is_verb2)):
        return False

    if w1.endswith('сь') and w2.endswith('сь'):
        return True

    return False


def is_short_full_adjective_match(word1: str, word2: str) -> bool:
    """Проверяет краткое↔полное прилагательное."""
    if not HAS_PYMORPHY:
        return False

    info1 = get_word_info(word1)
    info2 = get_word_info(word2)
    lemma1, pos1, _, _, _ = info1
    lemma2, pos2, _, _, _ = info2

    if lemma1 != lemma2:
        return False

    if (pos1 == 'ADJS' and pos2 == 'ADJF') or (pos1 == 'ADJF' and pos2 == 'ADJS'):
        return True

    return False


def is_lemma_match(word1: str, word2: str) -> bool:
    """Проверяет формы одного слова с учётом POS.

    ВАЖНО: Не фильтруем, если:
    - разный падеж для существительных/прилагательных
    - разное время для глаголов
    Это реальные ошибки чтеца!
    """
    if not HAS_PYMORPHY:
        return False

    p1 = morph.parse(word1)[0]
    p2 = morph.parse(word2)[0]

    lemma1 = p1.normal_form
    lemma2 = p2.normal_form
    pos1 = p1.tag.POS
    pos2 = p2.tag.POS

    if lemma1 != lemma2:
        return False

    # Деепричастия — не фильтруем переход в другую форму
    if pos1 == 'GRND' or pos2 == 'GRND':
        if pos1 != pos2:
            return False

    # Причастия — не фильтруем переход в другую форму
    participles = {'PRTF', 'PRTS'}
    if (pos1 in participles) != (pos2 in participles):
        return False

    # Число — разное число не фильтруем
    num1 = p1.tag.number
    num2 = p2.tag.number
    if num1 and num2 and num1 != num2:
        return False

    # ПРИЛАГАТЕЛЬНЫЕ: проверяем падеж
    # "верный" (им.) vs "верному" (дат.) — разные падежи, реальная ошибка!
    adjectives = {'ADJF', 'ADJS'}
    if pos1 in adjectives and pos2 in adjectives:
        case1 = p1.tag.case
        case2 = p2.tag.case
        if case1 and case2 and case1 != case2:
            # Безопасные пары падежей (звучат одинаково)
            safe_pairs = {frozenset({'nomn', 'accs'})}  # им. и вин. для неодуш.
            if frozenset({case1, case2}) not in safe_pairs:
                return False  # Разные падежи — не фильтруем!
        return True

    # ГЛАГОЛЫ: проверяем время
    # "получилось" (прош.) vs "получится" (буд.) — разное время, реальная ошибка!
    verbs = {'VERB', 'INFN'}
    if pos1 in verbs and pos2 in verbs:
        tense1 = p1.tag.tense
        tense2 = p2.tag.tense
        if tense1 and tense2 and tense1 != tense2:
            return False  # Разное время — не фильтруем!
        return True

    # СУЩЕСТВИТЕЛЬНЫЕ: проверяем падеж
    # "возможности" (род.) vs "возможность" (им.) — разные падежи, реальная ошибка!
    if pos1 == 'NOUN' and pos2 == 'NOUN':
        case1 = p1.tag.case
        case2 = p2.tag.case
        if case1 and case2 and case1 != case2:
            # Безопасные пары падежей
            safe_pairs = {frozenset({'nomn', 'accs'})}
            if frozenset({case1, case2}) not in safe_pairs:
                return False  # Разные падежи — не фильтруем!

    return True


def is_similar_by_levenshtein(word1: str, word2: str, threshold: int = 2) -> bool:
    """Схожесть по Левенштейну с адаптивным порогом."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if w1 == w2:
        return True

    min_len = min(len(w1), len(w2))
    if min_len <= 3:
        threshold = 1
    elif min_len <= 5:
        threshold = min(threshold, 2)

    return levenshtein_distance(w1, w2) <= threshold


def is_yandex_typical_error(word1: str, word2: str) -> bool:
    """Типичная ошибка Яндекса."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)
    return (w1, w2) in YANDEX_TYPICAL_ERRORS or (w2, w1) in YANDEX_TYPICAL_ERRORS


def is_prefix_variant(word1: str, word2: str) -> bool:
    """Проверяет, является ли одно слово другим с добавленной/убранной приставкой."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if len(w1) < 3 or len(w2) < 3:
        return False

    short, long = (w1, w2) if len(w1) < len(w2) else (w2, w1)

    for prefix in YANDEX_PREFIX_ERRORS:
        if long.startswith(prefix) and long[len(prefix):] == short:
            if HAS_PYMORPHY:
                parsed = morph.parse(short)
                if parsed and parsed[0].score > 0.1:
                    return True
            else:
                if len(short) >= 4:
                    return True

    return False


def is_interjection(word: str) -> bool:
    """Проверяет междометие."""
    from .constants import INTERJECTIONS
    return normalize_word(word) in INTERJECTIONS
