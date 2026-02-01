"""
Alignment Filter v1.0 — Унифицированная фильтрация артефактов выравнивания.

Этот модуль объединяет логику фильтрации артефактов выравнивания из:
- rules/alignment.py — check_alignment_artifact
- engine.py — ALIGNMENT_ARTIFACTS_DEL, ALIGNMENT_ARTIFACTS_INS
- context_verifier.py — split_word_artifact

ТИПЫ АРТЕФАКТОВ:
1. Length mismatch — одно слово намного длиннее другого (подстрока)
2. Boundary artifacts — артефакты на границах сегментов (deletion/insertion)
3. Split word — слово разбито на части (insertion + substitution)
4. Merge word — слова слились в одно (deletion + substitution)

v1.0 (2026-01-31): Начальная версия — унификация существующей логики
"""

VERSION = '1.0.0'

from typing import Dict, Any, Tuple, List, Optional, Set

# Импорт морфологии
try:
    from .comparison import (
        normalize_word, get_lemma, get_pos, get_case,
        HAS_PYMORPHY,
    )
except ImportError:
    HAS_PYMORPHY = False
    normalize_word = lambda x: x.lower().strip() if x else ''
    get_lemma = None
    get_pos = None
    get_case = None


# =============================================================================
# КОНСТАНТЫ — артефакты выравнивания
# =============================================================================

# Короткие слова, которые часто являются артефактами deletion
ALIGNMENT_ARTIFACTS_DEL: Set[str] = {
    'о', 'в', 'к', 'у', 'с', 'а', 'и',
    'на', 'за', 'по', 'из', 'от', 'до', 'ко',
}

# Короткие слова, которые часто являются артефактами insertion
ALIGNMENT_ARTIFACTS_INS: Set[str] = {
    'о', 'в', 'к', 'у', 'с', 'а', 'и',
    'на', 'за', 'по', 'из', 'от', 'до', 'ко',
    'то', 'ну', 'вот', 'вон', 'ой',
}

# Слабые слова в начале предложения
SENTENCE_START_WEAK: Set[str] = {
    'и', 'а', 'но', 'да', 'же', 'ли', 'бы',
}

# Короткие слабые слова (союзы, частицы)
SHORT_WEAK_WORDS: Set[str] = {
    'а', 'и', 'о', 'у', 'я',
    'ну', 'то', 'же', 'ли', 'бы', 'не', 'ни',
}

# Слабые союзы
WEAK_CONJUNCTIONS: Set[str] = {
    'и', 'а', 'но', 'да', 'же', 'или', 'либо',
}

# Слова, которые часто вставляются из-за выравнивания
WEAK_INSERTIONS: Set[str] = {
    'ну', 'вот', 'вон', 'это', 'то', 'же',
}

# Служебные слова (не фильтруем как артефакты)
FUNCTION_WORDS: Set[str] = {
    'в', 'на', 'с', 'к', 'у', 'о', 'по', 'из', 'за', 'от', 'до',
    'для', 'при', 'про', 'без', 'над', 'под', 'перед', 'между',
}

# Частицы составных слов (кто-то, что-нибудь)
COMPOUND_PARTICLES: Set[str] = {
    'то', 'либо', 'нибудь', 'ка',
}

# Однобуквенные согласные — артефакты выравнивания
SINGLE_CONSONANT_ARTIFACTS: Set[str] = {
    'с', 'м', 'в', 'п', 'к', 'ф', 'х', 'э', 'б', 'г', 'д', 'ж', 'з', 'л', 'н', 'р', 'т', 'ц', 'ч', 'ш', 'щ',
}


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ ФИЛЬТРАЦИИ
# =============================================================================

def check_alignment_artifact_substitution(
    w1: str,
    w2: str,
    get_lemma_func=None,
    get_pos_func=None,
    get_case_func=None,
) -> Tuple[bool, str]:
    """
    Проверяет substitution на артефакт выравнивания.

    Критерии:
    1. Одно слово является подстрокой другого (разная длина)
    2. Одинаковая лемма (если доступен pymorphy)

    Args:
        w1: Первое слово (transcript)
        w2: Второе слово (original)
        get_lemma_func: Функция получения леммы
        get_pos_func: Функция получения POS
        get_case_func: Функция получения падежа

    Returns:
        (should_filter, reason)
    """
    if not w1 or not w2:
        return False, ''

    len1, len2 = len(w1), len(w2)

    # Минимальная разница длин для артефакта
    if abs(len1 - len2) < 2:
        return False, ''

    # Проверяем подстроку
    is_substring = w1 in w2 or w2 in w1

    if not is_substring:
        return False, ''

    # Проверяем леммы — если разные, это может быть реальная ошибка
    if get_lemma_func:
        lemma1 = get_lemma_func(w1)
        lemma2 = get_lemma_func(w2)

        if lemma1 and lemma2 and lemma1 != lemma2:
            # Разные леммы — скорее всего реальная ошибка
            return False, ''

    # Определяем тип артефакта
    if w1 in w2:
        return True, f'alignment_artifact_length:{w1}⊂{w2}'
    else:
        return True, f'alignment_artifact_length:{w2}⊂{w1}'


def check_alignment_artifact_deletion(
    word: str,
    context: str = '',
    marker_pos: int = -1,
) -> Tuple[bool, str]:
    """
    Проверяет deletion на артефакт выравнивания.

    Критерии:
    1. Слово в списке известных артефактов
    2. Слово короткое и слабое (союз, частица)
    3. Слово в начале предложения (после точки)

    Args:
        word: Удалённое слово
        context: Контекст из оригинала
        marker_pos: Позиция маркера в контексте

    Returns:
        (should_filter, reason)
    """
    if not word:
        return False, ''

    word_lower = word.lower().strip()

    # 1. Известные артефакты deletion
    if word_lower in ALIGNMENT_ARTIFACTS_DEL:
        return True, 'alignment_artifact_del'

    # 2. Однобуквенные согласные
    if len(word_lower) == 1 and word_lower in SINGLE_CONSONANT_ARTIFACTS:
        return True, 'single_consonant_artifact'

    # 3. Короткие слабые слова (не в начале предложения)
    if word_lower in SHORT_WEAK_WORDS:
        if marker_pos > 0 and context:
            before_context = context[:marker_pos].rstrip()
            if before_context and before_context[-1] not in '.!?':
                return True, 'alignment_artifact_weak'

    # 4. Слабые союзы (не в начале предложения)
    if word_lower in WEAK_CONJUNCTIONS:
        if marker_pos > 0 and context:
            before_context = context[:marker_pos].rstrip()
            if before_context and before_context[-1] not in '.!?':
                return True, 'alignment_artifact_conj'
        elif marker_pos == -1:
            # Нет информации о позиции — консервативно фильтруем
            return True, 'alignment_artifact_conj'

    return False, ''


def check_alignment_artifact_insertion(
    word: str,
    context: str = '',
    marker_pos: int = -1,
) -> Tuple[bool, str]:
    """
    Проверяет insertion на артефакт выравнивания.

    Критерии:
    1. Слово в списке известных артефактов
    2. Однобуквенные согласные
    3. Слабые вставки (ну, вот, это)
    4. Союз после конца предложения

    Args:
        word: Вставленное слово
        context: Контекст из оригинала
        marker_pos: Позиция маркера в контексте

    Returns:
        (should_filter, reason)
    """
    if not word:
        return False, ''

    word_lower = word.lower().strip()

    # 1. Известные артефакты insertion
    if word_lower in ALIGNMENT_ARTIFACTS_INS:
        return True, 'alignment_artifact_ins'

    # 2. Однобуквенные согласные
    if len(word_lower) == 1 and word_lower in SINGLE_CONSONANT_ARTIFACTS:
        return True, 'single_consonant_artifact'

    # 3. Слабые вставки
    if word_lower in WEAK_INSERTIONS:
        return True, 'alignment_artifact_weak_ins'

    # 4. Союз и/а после конца предложения — это начало нового предложения
    if word_lower in {'и', 'а'}:
        if marker_pos > 0 and context:
            before = context[:marker_pos].rstrip()
            if before and before[-1] in '.!?':
                return True, 'sentence_start_conjunction'

    return False, ''


def check_split_word_artifact(
    inserted_word: str,
    transcript_context: str,
    original_context: str,
) -> Tuple[bool, str]:
    """
    Проверяет, является ли insertion артефактом разбиения слова.

    Паттерн: Яндекс разбивает слово на части
    - "ибахару" → "и" + "бахару"
    - "отнорок" → "от" + "норок"

    Args:
        inserted_word: Вставленное слово
        transcript_context: Контекст из транскрипции
        original_context: Контекст из оригинала

    Returns:
        (should_filter, reason)
    """
    if not inserted_word or not original_context:
        return False, ''

    inserted_lower = inserted_word.lower().strip()

    if len(inserted_lower) < 1:
        return False, ''

    # Нормализуем контексты
    orig_words = _normalize_context(original_context)
    trans_words = _normalize_context(transcript_context) if transcript_context else []

    if not orig_words:
        return False, ''

    # Ищем позиции вставленного слова в транскрипции
    inserted_positions = [i for i, w in enumerate(trans_words) if w == inserted_lower]

    MAX_DISTANCE = 2  # Максимальное расстояние для проверки "рядом"

    for i, orig_word in enumerate(orig_words):
        # Проверяем: оригинальное слово начинается с inserted?
        # Например: "ибахару" начинается с "и"
        if orig_word.startswith(inserted_lower) and len(orig_word) > len(inserted_lower):
            suffix = orig_word[len(inserted_lower):]
            # Проверяем, что суффикс находится РЯДОМ с inserted в транскрипции
            for ins_pos in inserted_positions:
                for offset in range(1, MAX_DISTANCE + 1):
                    neighbor_pos = ins_pos + offset
                    if neighbor_pos < len(trans_words) and trans_words[neighbor_pos] == suffix:
                        return True, f'split_word_artifact:{inserted_lower}+{suffix}={orig_word}'

        # Проверяем: оригинальное слово заканчивается на inserted?
        # Например: "бахаруи" заканчивается на "и"
        if orig_word.endswith(inserted_lower) and len(orig_word) > len(inserted_lower):
            prefix = orig_word[:-len(inserted_lower)]
            for ins_pos in inserted_positions:
                for offset in range(1, MAX_DISTANCE + 1):
                    neighbor_pos = ins_pos - offset
                    if neighbor_pos >= 0 and trans_words[neighbor_pos] == prefix:
                        return True, f'split_word_artifact:{prefix}+{inserted_lower}={orig_word}'

    return False, ''


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _normalize_context(context: str) -> List[str]:
    """
    Нормализует контекст: убирает пунктуацию, приводит к lower, разбивает на слова.
    """
    import re
    if not context:
        return []
    # Убираем пунктуацию, оставляем только буквы и пробелы
    cleaned = re.sub(r'[^\w\s]', ' ', context.lower())
    # Разбиваем на слова, убираем пустые
    words = [w.strip() for w in cleaned.split() if w.strip()]
    return words


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def check_alignment_artifact(
    error: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Главная функция — проверяет ошибку на артефакт выравнивания.

    Args:
        error: Словарь с ошибкой (type, wrong, correct, context, etc.)

    Returns:
        (should_filter, reason)
    """
    error_type = error.get('type', '')
    context = error.get('context', '')
    transcript_context = error.get('transcript_context', '')
    marker_pos = error.get('marker_pos', -1)

    if error_type == 'substitution':
        w1 = normalize_word(error.get('wrong', '') or error.get('transcript', ''))
        w2 = normalize_word(error.get('correct', '') or error.get('original', ''))
        return check_alignment_artifact_substitution(
            w1, w2,
            get_lemma_func=get_lemma if HAS_PYMORPHY else None,
            get_pos_func=get_pos if HAS_PYMORPHY else None,
            get_case_func=get_case if HAS_PYMORPHY else None,
        )

    elif error_type == 'deletion':
        word = normalize_word(error.get('correct', '') or error.get('original', ''))
        return check_alignment_artifact_deletion(word, context, marker_pos)

    elif error_type == 'insertion':
        word = normalize_word(error.get('wrong', '') or error.get('transcript', ''))

        # Сначала проверяем простые артефакты
        is_artifact, reason = check_alignment_artifact_insertion(word, context, marker_pos)
        if is_artifact:
            return True, reason

        # Затем проверяем split_word
        is_split, split_reason = check_split_word_artifact(word, transcript_context, context)
        if is_split:
            return True, split_reason

        return False, ''

    return False, ''


def get_alignment_filter_stats() -> Dict[str, Any]:
    """Возвращает статистику модуля."""
    return {
        'version': VERSION,
        'artifacts_del': len(ALIGNMENT_ARTIFACTS_DEL),
        'artifacts_ins': len(ALIGNMENT_ARTIFACTS_INS),
        'weak_words': len(SHORT_WEAK_WORDS),
        'conjunctions': len(WEAK_CONJUNCTIONS),
        'has_pymorphy': HAS_PYMORPHY,
    }
