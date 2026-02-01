"""
Экстракторы данных из ошибок v1.0.

Централизованное извлечение слов и контекста из словарей ошибок.
Устраняет дублирование кода извлечения в engine.py.

v1.0 (2026-01-31): Начальная версия
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

VERSION = '1.0.0'


@dataclass
class ExtractedWords:
    """Результат извлечения слов из ошибки."""
    word1: str  # transcript/wrong
    word2: str  # original/correct
    words: List[str]  # Все слова как список
    words_norm: List[str]  # Нормализованные слова

    @property
    def has_both(self) -> bool:
        """Есть ли оба слова (для substitution)."""
        return bool(self.word1 and self.word2)

    @property
    def primary(self) -> str:
        """Основное слово (для insertion/deletion)."""
        return self.word1 or self.word2


@dataclass
class ExtractedContext:
    """Результат извлечения контекста из ошибки."""
    context: str  # Контекст из оригинала
    transcript_context: str  # Контекст из транскрипции
    marker_pos: int  # Позиция маркера в контексте

    @property
    def has_context(self) -> bool:
        """Есть ли контекст."""
        return bool(self.context)

    @property
    def has_transcript_context(self) -> bool:
        """Есть ли контекст транскрипции."""
        return bool(self.transcript_context)


def extract_words(
    error: Dict[str, Any],
    normalize_func: Optional[callable] = None
) -> ExtractedWords:
    """
    Извлекает слова из ошибки в зависимости от типа.

    Args:
        error: Словарь с ошибкой
        normalize_func: Функция нормализации слов (опционально)

    Returns:
        ExtractedWords с word1, word2, words, words_norm

    Примеры:
        substitution: word1=transcript, word2=original
        insertion: word1=inserted_word, word2=''
        deletion: word1='', word2=deleted_word
    """
    error_type = error.get('type', '')

    if normalize_func is None:
        normalize_func = lambda x: x.lower().strip() if x else ''

    if error_type == 'substitution':
        word1 = error.get('wrong', '') or error.get('transcript', '')
        word2 = error.get('correct', '') or error.get('original', '')
        words = [word1, word2]

    elif error_type == 'insertion':
        word = error.get('wrong', '') or error.get('transcript', '') or error.get('word', '')
        word1, word2 = word, ''
        words = [word]

    elif error_type == 'deletion':
        word = error.get('correct', '') or error.get('original', '') or error.get('word', '')
        word1, word2 = '', word
        words = [word]

    else:
        # Неизвестный тип — пробуем извлечь что есть
        word = error.get('word', '') or error.get('wrong', '') or error.get('correct', '')
        word1 = word2 = word
        words = [word]

    words_norm = [normalize_func(w) for w in words]

    return ExtractedWords(
        word1=word1,
        word2=word2,
        words=words,
        words_norm=words_norm
    )


def extract_context(error: Dict[str, Any]) -> ExtractedContext:
    """
    Извлекает контекст из ошибки.

    Args:
        error: Словарь с ошибкой

    Returns:
        ExtractedContext с context, transcript_context, marker_pos
    """
    context = error.get('context', '')
    transcript_context = error.get('transcript_context', '')
    marker_pos = error.get('marker_pos', -1)

    return ExtractedContext(
        context=context,
        transcript_context=transcript_context,
        marker_pos=marker_pos
    )


def extract_all(
    error: Dict[str, Any],
    normalize_func: Optional[callable] = None
) -> Tuple[ExtractedWords, ExtractedContext]:
    """
    Извлекает и слова, и контекст из ошибки.

    Args:
        error: Словарь с ошибкой
        normalize_func: Функция нормализации слов

    Returns:
        Кортеж (ExtractedWords, ExtractedContext)
    """
    words = extract_words(error, normalize_func)
    context = extract_context(error)
    return words, context


def get_error_type(error: Dict[str, Any]) -> str:
    """Возвращает тип ошибки."""
    return error.get('type', '')


def get_time(error: Dict[str, Any]) -> float:
    """Возвращает временную метку ошибки."""
    return error.get('time', 0.0)


def is_merged_error(error: Dict[str, Any]) -> bool:
    """
    Проверяет, является ли ошибка результатом слияния ins+del.

    v9.5.1: Merged ошибки создаются merge_adjacent_ins_del()
    из соседних insertion+deletion.
    """
    return error.get('merged_from_ins_del', False)


def get_context_words(context: str, lower: bool = True) -> List[str]:
    """
    Разбивает контекст на слова.

    Args:
        context: Строка контекста
        lower: Приводить к lowercase

    Returns:
        Список слов
    """
    if not context:
        return []

    text = context.lower() if lower else context
    return text.split()


def find_word_in_context(
    word: str,
    context: str,
    lower: bool = True
) -> Tuple[bool, int]:
    """
    Ищет слово в контексте.

    Args:
        word: Искомое слово
        context: Строка контекста
        lower: Приводить к lowercase

    Returns:
        (найдено, позиция) — позиция -1 если не найдено
    """
    if not word or not context:
        return False, -1

    search_word = word.lower() if lower else word
    search_context = context.lower() if lower else context

    words = search_context.split()
    for i, w in enumerate(words):
        if w == search_word:
            return True, i

    return False, -1


# =========================================================================
# Утилиты для работы с позициями в контексте
# =========================================================================

def get_before_marker(context: str, marker_pos: int) -> str:
    """Возвращает текст до маркера."""
    if marker_pos <= 0 or not context:
        return ''
    return context[:marker_pos].rstrip()


def get_after_marker(context: str, marker_pos: int) -> str:
    """Возвращает текст после маркера."""
    if marker_pos < 0 or not context:
        return ''
    return context[marker_pos:].lstrip()


def ends_with_sentence(text: str) -> bool:
    """Проверяет, заканчивается ли текст концом предложения."""
    if not text:
        return False
    return text[-1] in '.!?'


def get_surrounding_words(
    words: List[str],
    position: int,
    window: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Возвращает слова до и после указанной позиции.

    Args:
        words: Список слов
        position: Позиция целевого слова
        window: Размер окна

    Returns:
        (слова_до, слова_после)
    """
    start = max(0, position - window)
    end = min(len(words), position + window + 1)

    before = words[start:position]
    after = words[position + 1:end]

    return before, after


if __name__ == '__main__':
    # Тесты
    print(f"Extractors v{VERSION}")
    print("=" * 40)

    # Тест substitution
    error1 = {
        'type': 'substitution',
        'wrong': 'услышал',
        'correct': 'услышав',
        'context': 'он услышав шум повернулся',
        'transcript_context': 'он услышал шум повернулся',
        'marker_pos': 3
    }
    words1 = extract_words(error1)
    print(f"substitution: word1='{words1.word1}', word2='{words1.word2}'")

    # Тест insertion
    error2 = {
        'type': 'insertion',
        'wrong': 'то',
        'context': 'как то там было',
        'transcript_context': 'как то там было',
    }
    words2 = extract_words(error2)
    print(f"insertion: word1='{words2.word1}', primary='{words2.primary}'")

    # Тест deletion
    error3 = {
        'type': 'deletion',
        'correct': 'же',
        'context': 'конечно же да',
    }
    words3 = extract_words(error3)
    print(f"deletion: word2='{words3.word2}', primary='{words3.primary}'")

    # Тест контекста
    ctx = extract_context(error1)
    print(f"context: has_context={ctx.has_context}, marker_pos={ctx.marker_pos}")

    # Тест before/after marker
    before = get_before_marker(error1['context'], error1['marker_pos'])
    print(f"before_marker: '{before}'")
