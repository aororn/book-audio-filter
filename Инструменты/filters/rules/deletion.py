"""
Правила фильтрации для deletion ошибок v1.1.

Мигрировано из engine.py.

Содержит:
- check_alignment_start_artifact — удаление в самом начале (time=0)
- check_character_name_unrecognized — удаление имени персонажа
- check_interjection_deletion — удаление междометия
- check_rare_adverb_deletion — удаление редкого наречия
- check_sentence_start_weak — удаление слабого слова в начале предложения
- check_hyphenated_part — удаление части дефисного слова
- check_compound_word_part — удаление части составного слова
- check_alignment_artifacts_del — удаление артефактов выравнивания
- check_short_weak_words — удаление коротких слабых слов
- check_weak_conjunctions — удаление слабых союзов

v1.1 (2026-01-31): Убраны fallback-блоки, прямые импорты
v1.0 (2026-01-31): Миграция из engine.py
"""

from typing import Dict, Any, Tuple, Set

VERSION = '1.1.0'

# Импорт констант (прямые импорты, без fallback)
from ..constants import (
    ALIGNMENT_ARTIFACTS_DEL,
    SHORT_WEAK_WORDS,
    WEAK_CONJUNCTIONS,
    SENTENCE_START_WEAK_WORDS,
    RARE_ADVERBS,
    INTERJECTIONS,
)

# Импорт детекторов и функций сравнения
from ..detectors import FULL_CHARACTER_NAMES, CHARACTER_NAMES_BASE
from ..comparison import normalize_word, is_interjection


def check_alignment_start_artifact(
    error_time: float
) -> Tuple[bool, str]:
    """
    Проверяет, является ли deletion артефактом в самом начале.

    Удаления с time=0 часто являются артефактами выравнивания.

    Args:
        error_time: Временная метка ошибки

    Returns:
        (should_filter, reason)
    """
    if error_time == 0:
        return True, 'alignment_start_artifact'
    return False, ''


def check_character_name_unrecognized(
    word: str,
    character_names: Set[str] = None,
    base_names: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удалённое слово именем персонажа.

    Яндекс часто не распознаёт имена персонажей.

    Args:
        word: Удалённое слово (нормализованное)
        character_names: Полный словарь имён (со склонениями)
        base_names: Базовые формы имён

    Returns:
        (should_filter, reason)
    """
    if len(word) < 3:
        return False, ''

    if character_names is None:
        character_names = FULL_CHARACTER_NAMES
    if base_names is None:
        base_names = CHARACTER_NAMES_BASE

    if word in base_names or word in character_names:
        return True, 'character_name_unrecognized'

    return False, ''


def check_interjection_deletion(
    word: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удалённое слово междометием.

    Args:
        word: Удалённое слово (нормализованное)

    Returns:
        (should_filter, reason)
    """
    if is_interjection(word):
        return True, 'interjection'
    return False, ''


def check_rare_adverb_deletion(
    word: str,
    rare_adverbs: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удалённое слово редким наречием.

    Яндекс часто не распознаёт редкие наречия.

    Args:
        word: Удалённое слово (нормализованное)
        rare_adverbs: Словарь редких наречий

    Returns:
        (should_filter, reason)
    """
    if rare_adverbs is None:
        rare_adverbs = RARE_ADVERBS

    if word in rare_adverbs:
        return True, 'rare_adverb'
    return False, ''


def check_sentence_start_weak(
    word: str,
    context: str,
    marker_pos: int,
    weak_words: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удаление слабого слова в начале предложения.

    После ./?/! слабые слова часто являются артефактами.

    Args:
        word: Удалённое слово (нормализованное)
        context: Контекст из оригинала
        marker_pos: Позиция маркера в контексте
        weak_words: Словарь слабых слов

    Returns:
        (should_filter, reason)
    """
    if weak_words is None:
        weak_words = SENTENCE_START_WEAK_WORDS

    if word not in weak_words:
        return False, ''

    if marker_pos > 0 and context:
        before_context = context[:marker_pos].rstrip()
        if before_context and before_context[-1] in '.!?':
            return True, 'sentence_start_weak'

    return False, ''


def check_hyphenated_part(
    word: str,
    context: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удаление частью дефисного слова.

    Пример: "тесь" от "Займи-тесь"

    Args:
        word: Удалённое слово (нормализованное)
        context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if len(word) < 2:
        return False, ''

    context_lower = context.lower() if context else ''

    # Паттерны: "-слово" или "слово-"
    if f'-{word}' in context_lower or f'{word}-' in context_lower:
        return True, 'hyphenated_part'

    return False, ''


def check_compound_word_part(
    word: str,
    context: str,
    transcript_context: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удаление частью составного слова.

    Примеры:
    - "возвышение" от "самовозвышение"
    - "звёздной" от "шестизвёздной"

    Args:
        word: Удалённое слово (нормализованное)
        context: Контекст из оригинала
        transcript_context: Контекст из транскрипции

    Returns:
        (should_filter, reason)
    """
    if len(word) < 4:
        return False, ''

    context_lower = context.lower() if context else ''
    transcript_ctx = transcript_context.lower() if transcript_context else ''

    # Объединяем слова из обоих контекстов
    ctx_words = context_lower.split() + transcript_ctx.split()

    for cw in ctx_words:
        cw_clean = normalize_word(cw)

        # Проверяем: cw заканчивается на word (само+возвышение)
        if len(cw_clean) > len(word) + 2 and cw_clean.endswith(word):
            prefix = cw_clean[:-len(word)]
            if len(prefix) >= 2:
                return True, 'compound_word_part'

        # Проверяем: cw начинается с word (звёздной+фракции)
        if len(cw_clean) > len(word) + 2 and cw_clean.startswith(word):
            suffix = cw_clean[len(word):]
            if len(suffix) >= 2:
                return True, 'compound_word_part'

    return False, ''


def check_alignment_artifacts_del(
    word: str,
    artifacts: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли слово артефактом выравнивания из словаря.

    Args:
        word: Удалённое слово (нормализованное)
        artifacts: Словарь артефактов

    Returns:
        (should_filter, reason)
    """
    if artifacts is None:
        artifacts = ALIGNMENT_ARTIFACTS_DEL

    if word in artifacts:
        return True, 'alignment_artifact'

    return False, ''


def check_short_weak_words(
    word: str,
    context: str,
    marker_pos: int,
    weak_words: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удаление короткого слабого слова артефактом.

    Фильтруем только если НЕ в начале предложения.

    Args:
        word: Удалённое слово (нормализованное)
        context: Контекст из оригинала
        marker_pos: Позиция маркера

    Returns:
        (should_filter, reason)
    """
    if weak_words is None:
        weak_words = SHORT_WEAK_WORDS

    if word not in weak_words:
        return False, ''

    if marker_pos > 0 and context:
        before_context = context[:marker_pos].rstrip()
        # Фильтруем только если НЕ после конца предложения
        if before_context and before_context[-1] not in '.!?':
            return True, 'alignment_artifact'

    return False, ''


def check_weak_conjunctions(
    word: str,
    context: str,
    marker_pos: int,
    conjunctions: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли удаление слабого союза артефактом.

    Args:
        word: Удалённое слово (нормализованное)
        context: Контекст из оригинала
        marker_pos: Позиция маркера

    Returns:
        (should_filter, reason)
    """
    if conjunctions is None:
        conjunctions = WEAK_CONJUNCTIONS

    if word not in conjunctions:
        return False, ''

    if marker_pos > 0 and context:
        before_context = context[:marker_pos].rstrip()
        if before_context and before_context[-1] not in '.!?':
            return True, 'alignment_artifact'
    elif marker_pos == 0:
        # В начале контекста — не фильтруем
        pass
    else:
        # marker_pos < 0 (не указан) — фильтруем
        return True, 'alignment_artifact'

    return False, ''


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def check_deletion_rules(
    error: Dict[str, Any],
    word_norm: str
) -> Tuple[bool, str]:
    """
    Применяет все правила для deletion ошибок.

    Args:
        error: Словарь с ошибкой
        word_norm: Нормализованное удалённое слово

    Returns:
        (should_filter, reason)
    """
    context = error.get('context', '')
    transcript_context = error.get('transcript_context', '')
    marker_pos = error.get('marker_pos', -1)
    error_time = error.get('time', 0)

    # 1. Артефакт в начале (time=0)
    result = check_alignment_start_artifact(error_time)
    if result[0]:
        return result

    # 2. Имя персонажа
    result = check_character_name_unrecognized(word_norm)
    if result[0]:
        return result

    # 3. Междометие
    result = check_interjection_deletion(word_norm)
    if result[0]:
        return result

    # 4. Редкое наречие
    result = check_rare_adverb_deletion(word_norm)
    if result[0]:
        return result

    # 5. Слабое слово в начале предложения
    result = check_sentence_start_weak(word_norm, context, marker_pos)
    if result[0]:
        return result

    # 6. Часть дефисного слова
    result = check_hyphenated_part(word_norm, context)
    if result[0]:
        return result

    # 7. Часть составного слова
    result = check_compound_word_part(word_norm, context, transcript_context)
    if result[0]:
        return result

    # 8. Артефакт из словаря
    result = check_alignment_artifacts_del(word_norm)
    if result[0]:
        return result

    # 9. Короткое слабое слово
    result = check_short_weak_words(word_norm, context, marker_pos)
    if result[0]:
        return result

    # 10. Слабый союз
    result = check_weak_conjunctions(word_norm, context, marker_pos)
    if result[0]:
        return result

    return False, ''


if __name__ == '__main__':
    print(f"Deletion Rules v{VERSION}")
    print("=" * 40)

    # Тесты
    test_cases = [
        # alignment_start_artifact
        ({'time': 0}, 'слово', True, 'alignment_start_artifact'),

        # interjection
        ({'time': 1.0, 'context': 'ах как хорошо'}, 'ах', True, 'interjection'),

        # rare_adverb
        ({'time': 1.0, 'context': 'так сказать эдак'}, 'эдак', True, 'rare_adverb'),

        # hyphenated_part
        ({'time': 1.0, 'context': 'Займи-тесь работой'}, 'тесь', True, 'hyphenated_part'),

        # sentence_start_weak
        ({'time': 1.0, 'context': 'Да. А потом ушёл', 'marker_pos': 4}, 'а', True, 'sentence_start_weak'),
    ]

    for error, word, expected, expected_reason in test_cases:
        result = check_deletion_rules(error, word)
        status = "✓" if result[0] == expected else "✗"
        print(f"{status} '{word}' → {result}")
