"""
Правила фильтрации для insertion ошибок v1.1.

Мигрировано из engine.py (строки 417-544).

Содержит:
- check_split_name_insertion — разбитое имя персонажа
- check_compound_particle_to — частица "то" от разбитого дефисного слова
- check_interrogative_split_to — "кто то", "что то" от дефисных слов
- check_split_suffix_insertion — суффикс разбитого слова
- check_split_word_fragment — двухбуквенный фрагмент разбитого слова
- check_yandex_split_insertions — вставки от разбитых Яндексом слов
- check_misrecognition_artifact — вставка похожа на слово в контексте
- check_unknown_word_artifact — вставка UNKN слова

v1.1 (2026-01-31): Убраны fallback-блоки, прямые импорты
v1.0 (2026-01-31): Миграция из engine.py
"""

import re
from typing import Dict, Any, Tuple, List, Set
from difflib import SequenceMatcher

VERSION = '1.2.0'

# Импорт констант (прямые импорты, без fallback)
from ..constants import (
    FUNCTION_WORDS,
    YANDEX_SPLIT_INSERTIONS,
    INTERROGATIVE_PRONOUNS,
    SKIP_SPLIT_FRAGMENT,
)

# Импорт детекторов и функций сравнения
from ..detectors import (
    FULL_CHARACTER_NAMES,
    is_split_name_insertion,
    is_split_compound_insertion,
)
from ..comparison import normalize_word, levenshtein_distance, HAS_PYMORPHY, morph


# Префиксы для compound_particle_to
COMPOUND_PREFIXES: List[str] = [
    'что', 'как', 'кто', 'где', 'когда', 'куда', 'откуда', 'почему', 'зачем',
    'какой', 'какая', 'какое', 'какие',
]

# Направления/глаголы после "то" — указывают на реальную вставку
DIRECTION_WORDS: Set[str] = {'туда', 'сюда', 'тут', 'здесь', 'теперь', 'тогда'}
VERB_ENDINGS: tuple = ('ся', 'ет', 'ит', 'ут', 'ат', 'ют', 'ёт')


def check_split_name_insertion(
    word: str,
    context: str,
    transcript_context: str,
    character_names: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли вставка частью разбитого имени персонажа.

    Пример: "шоу" + "гад" = "шаугат" (имя персонажа)

    Args:
        word: Вставленное слово (нормализованное)
        context: Контекст из оригинала
        transcript_context: Контекст из транскрипции
        character_names: Словарь имён персонажей

    Returns:
        (should_filter, reason)
    """
    if not word or len(word) < 2:
        return False, ''

    if character_names is None:
        character_names = FULL_CHARACTER_NAMES

    if not character_names:
        return False, ''

    # Проверяем с использованием встроенного детектора
    if is_split_name_insertion(word, transcript_context):
        return True, 'split_name'

    # Дополнительная проверка: комбинация с соседними словами
    if word not in FUNCTION_WORDS:
        transcript_ctx = transcript_context.lower() if transcript_context else ''
        if transcript_ctx:
            ctx_words = transcript_ctx.split()
            for i, ctx_word in enumerate(ctx_words):
                if ctx_word == word:
                    # Проверяем комбинацию с предыдущим словом
                    if i > 0:
                        prev_word = ctx_words[i - 1]
                        combined = prev_word + word
                        if combined in character_names:
                            return True, 'split_name_insertion'
                        for name in character_names:
                            if len(name) >= 6 and levenshtein_distance(combined, name) <= 1:
                                return True, 'split_name_insertion'

                    # Проверяем комбинацию со следующим словом
                    if i < len(ctx_words) - 1:
                        next_word = ctx_words[i + 1]
                        combined = word + next_word
                        if combined in character_names:
                            return True, 'split_name_insertion'
                        for name in character_names:
                            if len(name) >= 6 and levenshtein_distance(combined, name) <= 1:
                                return True, 'split_name_insertion'
                    break

    return False, ''


def check_interrogative_split_to(
    word: str,
    transcript_context: str,
    original_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS "то" как часть разбитого дефисного местоимения.

    Пример: "кто-то" → "кто то" (Яндекс разбил)

    Args:
        word: Вставленное слово
        transcript_context: Контекст из транскрипции
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if word != 'то':
        return False, ''

    transcript_ctx = transcript_context.lower() if transcript_context else ''
    original_ctx = original_context.lower() if original_context else ''

    for pronoun in INTERROGATIVE_PRONOUNS:
        pattern = f'{pronoun} то'
        if pattern in transcript_ctx:
            # Проверяем, что в оригинале ЕСТЬ дефисное слово
            hyphenated = f'{pronoun}-то'
            if hyphenated in original_ctx:
                return True, 'interrogative_split_to'

    return False, ''


def check_compound_particle_to(
    word: str,
    context: str,
    transcript_context: str = ''
) -> Tuple[bool, str]:
    """
    Проверяет INS "то" как часть разбитого составного слова.

    Примеры:
    - "как-то там" → "как то там" (фильтруем)
    - "что-то там" → "что то там" (фильтруем)
    - "кто сунется то туда" — НЕ фильтруем (реальная вставка)

    Args:
        word: Вставленное слово
        context: Контекст из оригинала
        transcript_context: Контекст из транскрипции

    Returns:
        (should_filter, reason)
    """
    if word != 'то':
        return False, ''

    context_lower = context.lower() if context else ''

    for prefix in COMPOUND_PREFIXES:
        # Используем regex с word boundaries
        pattern_regex = r'\b' + re.escape(prefix) + r'\s+то\b'
        match = re.search(pattern_regex, context_lower, re.IGNORECASE)

        if match:
            after_to_start = match.end()
            after_to = context_lower[after_to_start:].strip().split()

            if after_to:
                next_word = after_to[0]

                # Устойчивое выражение "{prefix}-то там" — фильтруем
                if next_word == 'там':
                    return True, 'compound_particle_to'

                # Если после "то" идёт направление или глагол — это реальная вставка
                if next_word in DIRECTION_WORDS or next_word.endswith(VERB_ENDINGS):
                    continue

            return True, 'compound_particle_to'

    return False, ''


def check_split_suffix_insertion(
    word: str,
    original_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS как суффикс разбитого слова.

    Пример: "говори" от "выторговали"

    Args:
        word: Вставленное слово (нормализованное)
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if len(word) < 4:
        return False, ''

    context_lower = original_context.lower() if original_context else ''
    ctx_words = context_lower.split()

    for ctx_word in ctx_words:
        ctx_clean = normalize_word(ctx_word)
        # Слово в контексте заканчивается на вставленное
        if len(ctx_clean) >= len(word) + 3 and ctx_clean.endswith(word):
            return True, 'split_suffix_insertion'

    return False, ''


def check_split_word_fragment(
    word: str,
    transcript_context: str,
    original_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS двухбуквенного фрагмента разбитого слова.

    Пример: "мы" от "големы" (голе + мы)

    Args:
        word: Вставленное слово (нормализованное)
        transcript_context: Контекст из транскрипции
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if len(word) != 2:
        return False, ''

    if word in SKIP_SPLIT_FRAGMENT:
        return False, ''

    transcript_ctx = transcript_context.lower() if transcript_context else ''
    original_ctx = original_context.lower() if original_context else ''

    trans_words = transcript_ctx.split()
    orig_words = original_ctx.split()

    for i, tw in enumerate(trans_words):
        if tw == word and i > 0:
            prev_trans = trans_words[i - 1]
            combined = prev_trans + word

            # Проверяем ТОЧНОЕ совпадение (без Левенштейна)
            for ow in orig_words:
                ow_clean = normalize_word(ow)
                if ow_clean == combined:
                    return True, 'split_word_fragment'
            break

    return False, ''


def check_yandex_split_insertions(
    word: str,
    transcript_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS из словаря разбитых Яндексом слов.

    Словарь YANDEX_SPLIT_INSERTIONS: {вставка: ожидаемое_предыдущее}

    Args:
        word: Вставленное слово (нормализованное)
        transcript_context: Контекст из транскрипции

    Returns:
        (should_filter, reason)
    """
    if word not in YANDEX_SPLIT_INSERTIONS:
        return False, ''

    expected_prev = YANDEX_SPLIT_INSERTIONS[word]
    transcript_ctx = transcript_context.lower() if transcript_context else ''
    pattern = f'{expected_prev} {word}'

    if pattern in transcript_ctx:
        return True, 'split_word_yandex'

    return False, ''


def check_misrecognition_artifact(
    word: str,
    original_context: str,
    threshold: float = 0.6
) -> Tuple[bool, str]:
    """
    Проверяет, похоже ли вставленное слово на слово в контексте.

    Примеры: "блядочное"~"ублюдочные", "оголим"~"големах"

    Args:
        word: Вставленное слово (нормализованное)
        original_context: Контекст из оригинала
        threshold: Порог сходства (0.0-1.0)

    Returns:
        (should_filter, reason)
    """
    if len(word) < 4:
        return False, ''

    context_lower = original_context.lower() if original_context else ''
    if not context_lower:
        return False, ''

    ctx_words = context_lower.split()

    for ctx_word in ctx_words:
        ctx_clean = ''.join(c for c in ctx_word if c.isalpha())
        # Пропускаем короткие и само вставленное слово
        if len(ctx_clean) >= 4 and ctx_clean != word:
            ratio = SequenceMatcher(None, word, ctx_clean).ratio()
            if ratio > threshold:
                return True, 'misrecognition_artifact'

    return False, ''


def check_unknown_word_artifact(
    word: str
) -> Tuple[bool, str]:
    """
    Проверяет, является ли слово неизвестным (UNKN) в pymorphy.

    Примеры: "бла" (обрыв "глава")

    Args:
        word: Вставленное слово (нормализованное)

    Returns:
        (should_filter, reason)
    """
    if not HAS_PYMORPHY or morph is None:
        return False, ''

    if len(word) < 2:
        return False, ''

    try:
        parsed = morph.parse(word)
        if parsed and 'UNKN' in str(parsed[0].tag):
            return True, 'unknown_word_artifact'
    except Exception:
        pass

    return False, ''


def check_split_compound_insertion(
    word: str,
    transcript_context: str,
    original_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS как часть разбитого составного слова.

    Использует встроенный детектор is_split_compound_insertion.

    Args:
        word: Вставленное слово (нормализованное)
        transcript_context: Контекст из транскрипции
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if is_split_compound_insertion(word, transcript_context, original_context):
        return True, 'split_compound'

    return False, ''


def check_split_word_insertion(
    word: str,
    original_context: str
) -> Tuple[bool, str]:
    """
    Проверяет INS как часть слова в контексте.

    Пример: "вор" как часть "говорит"

    Args:
        word: Вставленное слово (нормализованное)
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if len(word) < 3:
        return False, ''

    context_lower = original_context.lower() if original_context else ''
    context_words = context_lower.split()

    for ctx_word in context_words:
        ctx_clean = normalize_word(ctx_word)
        if len(ctx_clean) > len(word) + 2 and word in ctx_clean:
            return True, 'split_word_insertion'

    return False, ''


def check_context_name_artifact(
    word: str,
    context: str,
    transcript_context: str,
    character_names: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, является ли INS артефактом распознавания имени персонажа.

    Паттерн: Яндекс неправильно распознаёт имя персонажа и вставляет похожие
    по звучанию обычные слова.

    Примеры:
    - "кивнул род гош" → оригинал "кивнул Рутгош" (род = артефакт)
    - "тот год подхватил" → оригинал "Рутгош подхватил" (год = артефакт)
    - "туртакал... то такое" → оригинал "Тартакал...Тартакал" (то = артефакт)

    Два критерия фильтрации (OR):
    1. Фонетическое сходство: слово похоже на начало или часть имени (dist <= 2)
    2. Позиционное сходство: имя персонажа в оригинале на позиции ±2 от вставки

    Args:
        word: Вставленное слово (нормализованное)
        context: Контекст из оригинала
        transcript_context: Контекст из транскрипции
        character_names: Словарь имён персонажей

    Returns:
        (should_filter, reason)
    """
    if not word or len(word) < 2:
        return False, ''

    if character_names is None:
        character_names = FULL_CHARACTER_NAMES

    if not character_names:
        return False, ''

    context_lower = context.lower() if context else ''
    transcript_lower = transcript_context.lower() if transcript_context else ''

    if not context_lower or not transcript_lower:
        return False, ''

    # Очищаем контекст от пунктуации для сравнения позиций
    import re
    ctx_clean = re.sub(r'[^\w\s]', '', context_lower)
    trans_words = transcript_lower.split()
    ctx_words = ctx_clean.split()

    word_lower = word.lower()

    # Ищем позицию слова в транскрипте
    word_positions = [i for i, w in enumerate(trans_words) if w == word_lower]

    # Ищем позиции имён персонажей в оригинале
    name_positions = {}  # {имя: [позиции]}
    for i, cw in enumerate(ctx_words):
        for name in character_names:
            if len(name) >= 5 and name in cw:
                if name not in name_positions:
                    name_positions[name] = []
                name_positions[name].append(i)

    for word_pos in word_positions:
        # Проверяем: есть ли имя персонажа в оригинале в окрестности ±5 слов
        search_start = max(0, word_pos - 5)
        search_end = min(len(ctx_words), word_pos + 6)

        nearby_context = ' '.join(ctx_words[search_start:search_end])

        for name in character_names:
            # Пропускаем короткие имена (< 5 символов) — слишком много ложных срабатываний
            if len(name) < 5:
                continue

            if name in nearby_context:
                # Критерий 1: Фонетическое сходство с началом имени
                name_prefix = name[:len(word_lower)]
                dist = levenshtein_distance(word_lower, name_prefix)
                max_dist = 2 if len(word_lower) >= 3 else 1

                if dist <= max_dist:
                    return True, f'context_name_artifact:{name}'

                # Критерий 1b: Слово является частью имени
                if word_lower in name and len(word_lower) >= 3:
                    return True, f'context_name_artifact:{name}'

                # Критерий 2: Позиционное сходство
                # Имя в оригинале на позиции ±2 от вставки
                if name in name_positions:
                    for name_pos in name_positions[name]:
                        pos_diff = abs(word_pos - name_pos)
                        if pos_diff <= 2:
                            return True, f'context_name_artifact_pos:{name}(diff={pos_diff})'

    return False, ''


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def check_insertion_rules(
    error: Dict[str, Any],
    word_norm: str
) -> Tuple[bool, str]:
    """
    Применяет все правила для insertion ошибок.

    Args:
        error: Словарь с ошибкой
        word_norm: Нормализованное вставленное слово

    Returns:
        (should_filter, reason)
    """
    context = error.get('context', '')
    transcript_context = error.get('transcript_context', '')

    # 1. Разбитое имя персонажа
    result = check_split_name_insertion(
        word_norm, context, transcript_context
    )
    if result[0]:
        return result

    # 1.5. Артефакт имени персонажа в контексте (v1.2)
    # Пример: "род" рядом с "Рутгош" в оригинале
    result = check_context_name_artifact(
        word_norm, context, transcript_context
    )
    if result[0]:
        return result

    # 2. Частица "то" от дефисного местоимения
    result = check_interrogative_split_to(
        word_norm, transcript_context, context
    )
    if result[0]:
        return result

    # 3. Частица "то" от составного слова
    result = check_compound_particle_to(
        word_norm, context, transcript_context
    )
    if result[0]:
        return result

    # 4. Разбитое составное слово
    result = check_split_compound_insertion(
        word_norm, transcript_context, context
    )
    if result[0]:
        return result

    # 5. Разбитое Яндексом слово (словарь)
    result = check_yandex_split_insertions(
        word_norm, transcript_context
    )
    if result[0]:
        return result

    # 6. Суффикс разбитого слова
    result = check_split_suffix_insertion(
        word_norm, context
    )
    if result[0]:
        return result

    # 7. Двухбуквенный фрагмент
    result = check_split_word_fragment(
        word_norm, transcript_context, context
    )
    if result[0]:
        return result

    # 8. Misrecognition artifact
    result = check_misrecognition_artifact(
        word_norm, context
    )
    if result[0]:
        return result

    # 9. Unknown word artifact
    result = check_unknown_word_artifact(
        word_norm
    )
    if result[0]:
        return result

    # 10. Часть слова в контексте
    result = check_split_word_insertion(
        word_norm, context
    )
    if result[0]:
        return result

    return False, ''


if __name__ == '__main__':
    print(f"Insertion Rules v{VERSION}")
    print("=" * 40)

    # Тесты
    test_cases = [
        # compound_particle_to
        ('то', 'как то там было', '', True, 'compound_particle_to'),
        ('то', 'кто сунется то туда', '', False, ''),  # реальная вставка

        # interrogative_split_to
        ('то', 'кто то пришёл', 'кто-то пришёл', True, 'interrogative_split_to'),

        # split_word_fragment
        ('мы', 'голе мы шли', 'големы шли', True, 'split_word_fragment'),

        # skip_split_fragment
        ('и', 'голе и шли', 'големы шли', False, ''),  # "и" в SKIP_SPLIT_FRAGMENT
    ]

    for word, ctx, orig_ctx, expected, expected_reason in test_cases:
        if word == 'то' and 'кто-то' in orig_ctx:
            result = check_interrogative_split_to(word, ctx, orig_ctx)
        elif word == 'то':
            result = check_compound_particle_to(word, ctx)
        else:
            result = check_split_word_fragment(word, ctx, orig_ctx)

        status = "✓" if result[0] == expected else "✗"
        print(f"{status} '{word}' in '{ctx[:30]}...' → {result}")
