"""
SlidingWindow v1.1 — Фонетическое сравнение без пробелов.

Дополняет WindowVerifier. Выявляет артефакты выравнивания когда
слова правильные, но границы съехали из-за пробелов.

Пример:
    Оригинал:     "и сам он"
    Транскрипт:   "исамон"
    Без пробелов: "исамон" == "исамон" → артефакт выравнивания

Алгоритм:
    1. Склеить оригинал без пробелов
    2. Склеить транскрипт без пробелов
    3. Сравнить fuzz.ratio >= 95% → артефакт
    4. Также проверить подстроки (sliding window)

Использование:
    - При substitution/deletion/insertion проверяем контекст ±2 слова
    - Если склеенные версии совпадают — это артефакт, а не ошибка
    - Добавляет -100 в скоринг (обнуляет)

v1.1.0 (2026-01-31): Пороги из config.py
v1.0.0 (2026-01-30): Начальная версия
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass

VERSION = '1.1.0'
VERSION_DATE = '2026-01-31'

# v1.1: Пороги из config.py
from .config import get_phonetic_match_threshold, get_substring_match_threshold

# Алиасы для обратной совместимости
PHONETIC_MATCH_THRESHOLD = 95  # Используй get_phonetic_match_threshold()
SUBSTRING_MATCH_THRESHOLD = 90  # Используй get_substring_match_threshold()

# Размер окна контекста (слова слева + справа)
DEFAULT_WINDOW_SIZE = 2


@dataclass
class SlidingResult:
    """Результат проверки скользящим окном."""
    is_artifact: bool
    match_type: str  # 'exact', 'fuzzy', 'substring', 'none'
    similarity: int  # 0-100
    original_concat: str
    transcript_concat: str
    details: Optional[dict] = None


class SlidingWindow:
    """
    Скользящее окно для фонетического сравнения.

    Использование:
        sw = SlidingWindow()
        result = sw.check_artifact(
            original_words=['и', 'сам', 'он'],
            transcript_words=['исамон'],
            error_index=1  # позиция ошибки
        )
        if result.is_artifact:
            # пропустить как артефакт выравнивания
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self._fuzz_available = self._check_fuzz()

    def _check_fuzz(self) -> bool:
        """Проверить доступность rapidfuzz/thefuzz."""
        try:
            from rapidfuzz import fuzz
            return True
        except ImportError:
            try:
                from thefuzz import fuzz
                return True
            except ImportError:
                print("[SlidingWindow] rapidfuzz/thefuzz не установлен")
                return False

    def _get_fuzz_ratio(self, s1: str, s2: str) -> int:
        """Получить fuzz.ratio между строками."""
        if not self._fuzz_available:
            # Fallback: простое сравнение
            return 100 if s1 == s2 else 0

        try:
            from rapidfuzz import fuzz
        except ImportError:
            from thefuzz import fuzz

        return fuzz.ratio(s1, s2)

    def _clean_text(self, text: str) -> str:
        """Очистить текст: только буквы, нижний регистр."""
        return re.sub(r'[^а-яёa-z]', '', text.lower())

    def _concat_words(self, words: List[str]) -> str:
        """Склеить слова без пробелов."""
        return ''.join(self._clean_text(w) for w in words)

    def check_artifact(
        self,
        original_words: List[str],
        transcript_words: List[str],
        error_index: Optional[int] = None,
    ) -> SlidingResult:
        """
        Проверить, является ли разница артефактом выравнивания.

        Args:
            original_words: Слова оригинала
            transcript_words: Слова транскрипта
            error_index: Индекс ошибки в списке (для окна)

        Returns:
            SlidingResult с результатом проверки
        """
        # Если указан индекс, берём окно вокруг него
        if error_index is not None:
            start = max(0, error_index - self.window_size)
            end = min(len(original_words), error_index + self.window_size + 1)
            original_words = original_words[start:end]

            # Аналогично для транскрипта (может быть другой длины)
            t_start = max(0, error_index - self.window_size)
            t_end = min(len(transcript_words), error_index + self.window_size + 1)
            transcript_words = transcript_words[t_start:t_end]

        # Склеиваем
        orig_concat = self._concat_words(original_words)
        trans_concat = self._concat_words(transcript_words)

        # Пустые строки
        if not orig_concat or not trans_concat:
            return SlidingResult(
                is_artifact=False,
                match_type='none',
                similarity=0,
                original_concat=orig_concat,
                transcript_concat=trans_concat,
            )

        # 1. Точное совпадение
        if orig_concat == trans_concat:
            return SlidingResult(
                is_artifact=True,
                match_type='exact',
                similarity=100,
                original_concat=orig_concat,
                transcript_concat=trans_concat,
            )

        # 2. Fuzzy match
        ratio = self._get_fuzz_ratio(orig_concat, trans_concat)
        if ratio >= get_phonetic_match_threshold():
            return SlidingResult(
                is_artifact=True,
                match_type='fuzzy',
                similarity=ratio,
                original_concat=orig_concat,
                transcript_concat=trans_concat,
            )

        # 3. Проверка подстроки (один содержит другого)
        substring_result = self._check_substring(orig_concat, trans_concat)
        if substring_result:
            return substring_result

        # Не артефакт
        return SlidingResult(
            is_artifact=False,
            match_type='none',
            similarity=ratio,
            original_concat=orig_concat,
            transcript_concat=trans_concat,
        )

    def _check_substring(
        self,
        orig: str,
        trans: str,
    ) -> Optional[SlidingResult]:
        """Проверить, является ли один текст подстрокой другого."""
        # Оригинал содержит транскрипт
        if trans in orig:
            coverage = len(trans) / len(orig) * 100
            if coverage >= get_substring_match_threshold():
                return SlidingResult(
                    is_artifact=True,
                    match_type='substring',
                    similarity=int(coverage),
                    original_concat=orig,
                    transcript_concat=trans,
                    details={'direction': 'trans_in_orig', 'coverage': coverage},
                )

        # Транскрипт содержит оригинал
        if orig in trans:
            coverage = len(orig) / len(trans) * 100
            if coverage >= get_substring_match_threshold():
                return SlidingResult(
                    is_artifact=True,
                    match_type='substring',
                    similarity=int(coverage),
                    original_concat=orig,
                    transcript_concat=trans,
                    details={'direction': 'orig_in_trans', 'coverage': coverage},
                )

        return None

    def check_context_match(
        self,
        original_word: str,
        transcript_word: str,
        original_context: List[str],
        transcript_context: List[str],
    ) -> SlidingResult:
        """
        Проверить ошибку с учётом контекста.

        Более удобный метод: передаём само слово и его контекст отдельно.

        Args:
            original_word: Слово из оригинала (предполагаемая ошибка)
            transcript_word: Слово из транскрипта
            original_context: Контекст из оригинала [слова вокруг]
            transcript_context: Контекст из транскрипта

        Returns:
            SlidingResult
        """
        # Добавляем слово в контекст для сравнения
        orig_words = original_context + [original_word]
        trans_words = transcript_context + [transcript_word]

        return self.check_artifact(orig_words, trans_words)


# Глобальный экземпляр
_sliding_window: Optional[SlidingWindow] = None


def get_sliding_window() -> SlidingWindow:
    """Получить глобальный экземпляр SlidingWindow."""
    global _sliding_window
    if _sliding_window is None:
        _sliding_window = SlidingWindow()
    return _sliding_window


def is_alignment_artifact(
    original_words: List[str],
    transcript_words: List[str],
    error_index: Optional[int] = None,
) -> bool:
    """Удобная функция для проверки артефакта."""
    result = get_sliding_window().check_artifact(
        original_words, transcript_words, error_index
    )
    return result.is_artifact


def check_phonetic_match(
    original: str,
    transcript: str,
) -> Tuple[bool, int]:
    """
    Проверить фонетическое совпадение строк.

    Returns:
        (is_match, similarity)
    """
    sw = get_sliding_window()
    orig_clean = sw._clean_text(original)
    trans_clean = sw._clean_text(transcript)

    if orig_clean == trans_clean:
        return (True, 100)

    ratio = sw._get_fuzz_ratio(orig_clean, trans_clean)
    return (ratio >= get_phonetic_match_threshold(), ratio)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_sliding_window():
    """Тест SlidingWindow."""
    print('=' * 60)
    print('ТЕСТ: SlidingWindow')
    print('=' * 60)

    sw = SlidingWindow()

    # Тест 1: Точное совпадение без пробелов
    print('\n1. Склеенные слова ("и сам он" vs "исамон"):')
    result = sw.check_artifact(['и', 'сам', 'он'], ['исамон'])
    print(f'   is_artifact: {result.is_artifact}')
    print(f'   match_type: {result.match_type}')
    print(f'   similarity: {result.similarity}')
    print(f'   concat: "{result.original_concat}" vs "{result.transcript_concat}"')

    # Тест 2: Fuzzy match
    print('\n2. Похожие склейки ("способа" vs "выхода"):')
    result = sw.check_artifact(['способа'], ['выхода'])
    print(f'   is_artifact: {result.is_artifact}')
    print(f'   similarity: {result.similarity}')

    # Тест 3: С контекстом
    print('\n3. С контекстом ("искать способа" vs "искать выхода"):')
    result = sw.check_artifact(['искать', 'способа'], ['искать', 'выхода'])
    print(f'   is_artifact: {result.is_artifact}')
    print(f'   similarity: {result.similarity}')
    print(f'   concat: "{result.original_concat}" vs "{result.transcript_concat}"')

    # Тест 4: Подстрока
    print('\n4. Подстрока ("ничего" vs "ни"):')
    result = sw.check_artifact(['ничего'], ['ни'])
    print(f'   is_artifact: {result.is_artifact}')
    print(f'   match_type: {result.match_type}')
    print(f'   details: {result.details}')

    # Тест 5: Реальная ошибка
    print('\n5. Реальная ошибка ("ТЕРЯЮ" vs "теряя"):')
    result = sw.check_artifact(['ТЕРЯЮ'], ['теряя'])
    print(f'   is_artifact: {result.is_artifact}')
    print(f'   similarity: {result.similarity}')

    # Тест 6: check_phonetic_match
    print('\n6. check_phonetic_match:')
    test_cases = [
        ("и сам он", "исамон"),
        ("способа", "выхода"),
        ("получится", "получилось"),
    ]
    for orig, trans in test_cases:
        is_match, sim = check_phonetic_match(orig, trans)
        print(f'   "{orig}" vs "{trans}": match={is_match}, sim={sim}')

    print()
    print('=' * 60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('=' * 60)


if __name__ == '__main__':
    test_sliding_window()
