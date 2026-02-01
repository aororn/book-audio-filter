"""
Правила фильтрации для substitution ошибок v1.1.

Мигрировано из engine.py.

Содержит:
- check_yandex_merge_artifact — Яндекс слил слова (яХХ←я)
- check_yandex_truncate_artifact — Яндекс усёк слово (и←их)
- check_yandex_expand_artifact — Яндекс расширил слово (итак←и)
- check_weak_words_identical — одинаковые слабые слова
- check_weak_words_same_lemma — слабые слова с одинаковой леммой
- check_sentence_start_conjunction — союз в начале предложения
- check_identical_normalized — идентичные после нормализации
- check_compound_word — составное слово
- check_merged_word — склеенное слово
- check_case_form — падежная форма
- check_adverb_adjective — наречие/прилагательное
- check_short_full_adjective — краткое/полное прилагательное
- check_verb_gerund_safe — глагол/деепричастие

v1.1 (2026-01-31): Убраны fallback-блоки, прямые импорты
v1.0 (2026-01-31): Миграция из engine.py
"""

from typing import Dict, Any, Tuple, Set, Optional

VERSION = '1.1.0'

# Импорт констант (прямые импорты, без fallback)
from ..constants import WEAK_WORDS, ALIGNMENT_ARTIFACTS_INS, WEAK_INSERTIONS

# Импорт функций сравнения
from ..comparison import (
    normalize_word, get_lemma, is_lemma_match, is_homophone_match,
    is_case_form_match, is_adverb_adjective_match,
    is_short_full_adjective_match, is_verb_gerund_safe_match,
    is_yandex_typical_error, is_yandex_name_error,
    HAS_PYMORPHY,
)
from ..detectors import is_compound_word_match, is_merged_word_error


def check_yandex_merge_artifact(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, слил ли Яндекс "я" со следующим словом.

    Паттерн: яХХ←я (Яндекс слил "я" со следующим словом)
    Пример: оригинал "Я же" → транскрипт "яша"

    Args:
        w1: transcript (что Яндекс услышал)
        w2: original (что должно быть)

    Returns:
        (should_filter, reason)
    """
    if len(w1) > 1 and w2 == 'я' and w1.startswith('я'):
        return True, 'yandex_merge_artifact'
    return False, ''


def check_yandex_truncate_artifact(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, усёк ли Яндекс многобуквенное слово до однобуквенного.

    Паттерн: и←их (Яндекс усёк)
    Пример: оригинал "Их главу" → транскрипт "И главу"

    Args:
        w1: transcript (что Яндекс услышал)
        w2: original (что должно быть)

    Returns:
        (should_filter, reason)
    """
    if w1 in {'и', 'а', 'я', 'е'} and len(w2) > 1 and w2.startswith(w1):
        return True, 'yandex_truncate_artifact'
    return False, ''


def check_yandex_expand_artifact(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, расширил ли Яндекс однобуквенное слово.

    Паттерн: итак←и (Яндекс расширил)
    Пример: оригинал "И так" → транскрипт "итак"

    ИСКЛЮЧЕНИЕ: 'или' — это реальная ошибка чтеца (Golden)

    Args:
        w1: transcript (что Яндекс услышал)
        w2: original (что должно быть)

    Returns:
        (should_filter, reason)
    """
    # ИСКЛЮЧЕНИЕ: 'или' — реальная ошибка (Golden 30:38)
    if w1 == 'или':
        return False, ''

    if w2 in {'и', 'а'} and len(w1) > 2 and w1.startswith(w2):
        return True, 'yandex_expand_artifact'
    return False, ''


def check_weak_words_identical(
    w1: str,
    w2: str,
    weak_words: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли оба слова слабыми и идентичными.

    Args:
        w1: transcript
        w2: original
        weak_words: Словарь слабых слов

    Returns:
        (should_filter, reason)
    """
    if weak_words is None:
        weak_words = WEAK_WORDS

    if w1 in weak_words and w2 in weak_words:
        if w1 == w2:
            return True, 'weak_words_identical'
    return False, ''


def check_weak_words_same_lemma(
    w1: str,
    w2: str,
    weak_words: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли оба слабых слова формами одной леммы.

    Args:
        w1: transcript
        w2: original
        weak_words: Словарь слабых слов

    Returns:
        (should_filter, reason)
    """
    if weak_words is None:
        weak_words = WEAK_WORDS

    if w1 in weak_words and w2 in weak_words:
        if HAS_PYMORPHY and is_lemma_match(w1, w2):
            return True, 'weak_words_same_lemma'
    return False, ''


def check_sentence_start_conjunction(
    word: str,
    context: str,
    marker_pos: int
) -> Tuple[bool, str]:
    """
    Проверяет insertion и/а после конца предложения.

    "довольно. И так", "Да. Они" — союз в начале предложения норма.

    Args:
        word: Слово (нормализованное)
        context: Контекст из оригинала
        marker_pos: Позиция маркера

    Returns:
        (should_filter, reason)
    """
    if word not in {'и', 'а'}:
        return False, ''

    if marker_pos > 0 and context:
        before = context[:marker_pos].rstrip()
        if before and before[-1] in '.!?':
            return True, 'sentence_start_conjunction'

    return False, ''


def check_identical_normalized(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, идентичны ли слова после нормализации.

    Args:
        w1: transcript (нормализованное)
        w2: original (нормализованное)

    Returns:
        (should_filter, reason)
    """
    if w1 == w2:
        return True, 'identical_normalized'
    return False, ''


def check_homophone(
    w1: str,
    w2: str,
    use_homophones: bool = True
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова омофонами.

    Args:
        w1: transcript
        w2: original
        use_homophones: Использовать ли проверку омофонов

    Returns:
        (should_filter, reason)
    """
    if use_homophones and is_homophone_match(w1, w2):
        return True, 'homophone'
    return False, ''


def check_compound_word(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова формами составного слова.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_compound_word_match(w1, w2):
        return True, 'compound_word'
    return False, ''


def check_merged_word(
    word: str,
    context: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли слово результатом склейки.

    Args:
        word: transcript
        context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if is_merged_word_error(word, context):
        return True, 'merged_word'
    return False, ''


def check_case_form(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова падежными формами.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_case_form_match(w1, w2):
        return True, 'case_form'
    return False, ''


def check_adverb_adjective(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова наречием/прилагательным.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_adverb_adjective_match(w1, w2):
        return True, 'adverb_adjective'
    return False, ''


def check_short_full_adjective(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова краткой/полной формой прилагательного.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_short_full_adjective_match(w1, w2):
        return True, 'short_full_adjective'
    return False, ''


def check_verb_gerund_safe(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, являются ли слова безопасной парой глагол/деепричастие.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_verb_gerund_safe_match(w1, w2):
        return True, 'verb_gerund_safe'
    return False, ''


def check_yandex_typical(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли пара типичной ошибкой Яндекса.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_yandex_typical_error(w1, w2):
        return True, 'yandex_typical'
    return False, ''


def check_yandex_name(
    w1: str,
    w2: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли пара ошибкой Яндекса на имени.

    Args:
        w1: transcript
        w2: original

    Returns:
        (should_filter, reason)
    """
    if is_yandex_name_error(w1, w2):
        return True, 'yandex_name_error'
    return False, ''


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def check_substitution_rules(
    error: Dict[str, Any],
    w1: str,
    w2: str,
    use_homophones: bool = True,
    weak_words: Set[str] = None
) -> Tuple[bool, str]:
    """
    Применяет правила для substitution ошибок уровня 2-3.

    НЕ включает:
    - Уровень -1 (HARD_NEGATIVES) — protection.py
    - Уровень -0.6 (interjection_pair) — engine.py
    - Уровень -0.55 (yandex_phonetic_pair) — phonetics.py
    - Уровень -0.5 (semantic_slip) — protection.py
    - Уровень 0 (morpho_rules) — morpho_rules.py
    - Уровень 0.3 (safe_ending) — alignment.py
    - Уровень 0.6 (alignment_artifact) — alignment.py
    - Уровень 10 (ML) — ml_classifier.py
    - Уровень 12 (context_verifier) — context_verifier.py

    Args:
        error: Словарь с ошибкой
        w1: transcript (нормализованное)
        w2: original (нормализованное)
        use_homophones: Использовать ли проверку омофонов
        weak_words: Словарь слабых слов

    Returns:
        (should_filter, reason)
    """
    context = error.get('context', '')
    marker_pos = error.get('marker_pos', -1)

    if weak_words is None:
        weak_words = WEAK_WORDS

    # 1. Идентичные после нормализации
    result = check_identical_normalized(w1, w2)
    if result[0]:
        return result

    # 2. Омофоны
    result = check_homophone(w1, w2, use_homophones)
    if result[0]:
        return result

    # 3. Составное слово
    result = check_compound_word(w1, w2)
    if result[0]:
        return result

    # 4. Склеенное слово
    result = check_merged_word(w1, context)
    if result[0]:
        return result

    # 5. Падежная форма
    result = check_case_form(w1, w2)
    if result[0]:
        return result

    # 6. Наречие/прилагательное
    result = check_adverb_adjective(w1, w2)
    if result[0]:
        return result

    # 7. Краткое/полное прилагательное
    result = check_short_full_adjective(w1, w2)
    if result[0]:
        return result

    # 8. Глагол/деепричастие
    result = check_verb_gerund_safe(w1, w2)
    if result[0]:
        return result

    # 9. Типичная ошибка Яндекса
    result = check_yandex_typical(w1, w2)
    if result[0]:
        return result

    # 10. Ошибка Яндекса на имени
    result = check_yandex_name(w1, w2)
    if result[0]:
        return result

    # 11. Слабые слова одинаковые
    result = check_weak_words_identical(w1, w2, weak_words)
    if result[0]:
        return result

    # 12. Слабые слова одинаковая лемма
    result = check_weak_words_same_lemma(w1, w2, weak_words)
    if result[0]:
        return result

    # 13. Yandex merge artifact
    result = check_yandex_merge_artifact(w1, w2)
    if result[0]:
        return result

    # 14. Yandex truncate artifact
    result = check_yandex_truncate_artifact(w1, w2)
    if result[0]:
        return result

    # 15. Yandex expand artifact
    result = check_yandex_expand_artifact(w1, w2)
    if result[0]:
        return result

    return False, ''


if __name__ == '__main__':
    print(f"Substitution Rules v{VERSION}")
    print("=" * 40)

    # Тесты
    test_cases = [
        # identical_normalized
        ('слово', 'слово', True, 'identical_normalized'),

        # yandex_merge_artifact
        ('яша', 'я', True, 'yandex_merge_artifact'),

        # yandex_truncate_artifact
        ('и', 'их', True, 'yandex_truncate_artifact'),

        # yandex_expand_artifact
        ('итак', 'и', True, 'yandex_expand_artifact'),

        # yandex_expand_artifact — исключение для 'или'
        ('или', 'и', False, ''),
    ]

    for w1, w2, expected, expected_reason in test_cases:
        error = {'context': ''}
        result = check_substitution_rules(error, w1, w2)
        status = "✓" if result[0] == expected else "✗"
        print(f"{status} '{w1}' → '{w2}' = {result}")
