"""
WindowVerifier v1.0 — Скользящее окно для верификации сегментов.

Используется для:
1. Выявления технического шума (перестановки слов, пунктуация)
2. Защиты от скрытия реальных ошибок
3. Финальной проверки перед показом пользователю

v1.0 (2026-01-29): Начальная версия — консервативный подход
"""

VERSION = '1.1.0'
VERSION_DATE = '2026-01-29'

from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum
import re


# =============================================================================
# СТАТУСЫ ВЕРИФИКАЦИИ
# =============================================================================

class VerificationStatus(Enum):
    """Результат верификации сегмента."""
    TECHNICAL_OK = "technical_ok"    # Технический шум, можно скрыть
    SUSPICIOUS = "suspicious"         # Требует проверки человеком
    ERROR = "error"                   # Явная ошибка чтеца


@dataclass
class VerificationResult:
    """Результат верификации."""
    status: VerificationStatus
    reason: str
    similarity: float
    details: dict = None


# =============================================================================
# WINDOW VERIFIER
# =============================================================================

class WindowVerifier:
    """
    Проверяет сегменты текста на технический шум vs реальные ошибки.

    Принцип: КОНСЕРВАТИВНЫЙ
    - Если есть сомнения — показываем человеку (SUSPICIOUS)
    - Скрываем только 100% технический шум (TECHNICAL_OK)
    - Реальные ошибки помечаем как ERROR
    """

    # Пороги
    MIN_SEGMENT_LENGTH = 15          # Минимальная длина для анализа
    TECHNICAL_OK_THRESHOLD = 0.95    # >= 95% схожесть = технический шум
    ERROR_THRESHOLD = 0.70           # < 70% = явная ошибка
    MAX_LENGTH_DIFF_RATIO = 0.20     # Максимальная разница длин (20%)
    MAX_WORD_LOSS = 1                # Максимум потерянных слов для TECHNICAL_OK

    def __init__(self):
        pass

    def verify_segment(
        self,
        author_segment: str,
        audio_segment: str
    ) -> VerificationResult:
        """
        Верифицирует сегмент: технический шум или реальная ошибка?

        Args:
            author_segment: текст из оригинала
            audio_segment: текст из транскрипции

        Returns:
            VerificationResult
        """
        # Очищаем для сравнения
        author_clean = self.clean_for_comparison(author_segment)
        audio_clean = self.clean_for_comparison(audio_segment)

        # Проверка минимальной длины
        if len(author_clean) < self.MIN_SEGMENT_LENGTH:
            return VerificationResult(
                status=VerificationStatus.SUSPICIOUS,
                reason=f"сегмент_слишком_короткий ({len(author_clean)} < {self.MIN_SEGMENT_LENGTH})",
                similarity=0.0,
                details={'author_len': len(author_clean), 'audio_len': len(audio_clean)}
            )

        # Проверка разницы длин
        length_ratio = self._calculate_length_ratio(author_clean, audio_clean)
        if length_ratio > self.MAX_LENGTH_DIFF_RATIO:
            return VerificationResult(
                status=VerificationStatus.SUSPICIOUS,
                reason=f"разница_длин ({length_ratio:.1%} > {self.MAX_LENGTH_DIFF_RATIO:.0%})",
                similarity=0.0,
                details={'length_ratio': length_ratio}
            )

        # Подсчёт потерянных слов
        word_loss = self._count_word_loss(author_segment, audio_segment)
        if word_loss > self.MAX_WORD_LOSS:
            return VerificationResult(
                status=VerificationStatus.SUSPICIOUS,
                reason=f"потеряно_{word_loss}_слов (>{self.MAX_WORD_LOSS})",
                similarity=0.0,
                details={'word_loss': word_loss}
            )

        # Вычисляем схожесть
        similarity = self.get_levenshtein_similarity(author_clean, audio_clean)

        # v1.1: Проверяем, есть ли слова которые РЕАЛЬНО отличаются буквами
        # Если да — это НЕ технический шум, даже при высокой схожести контекста
        different_words = self._find_different_words(author_segment, audio_segment)
        if different_words:
            return VerificationResult(
                status=VerificationStatus.SUSPICIOUS,
                reason=f"слова_отличаются: {different_words[0][0]}≠{different_words[0][1]}",
                similarity=similarity,
                details={'different_words': different_words, 'word_loss': word_loss}
            )

        # Определяем статус
        if similarity >= self.TECHNICAL_OK_THRESHOLD:
            return VerificationResult(
                status=VerificationStatus.TECHNICAL_OK,
                reason=f"высокая_схожесть ({similarity:.1%} >= {self.TECHNICAL_OK_THRESHOLD:.0%})",
                similarity=similarity,
                details={'word_loss': word_loss}
            )
        elif similarity < self.ERROR_THRESHOLD:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                reason=f"низкая_схожесть ({similarity:.1%} < {self.ERROR_THRESHOLD:.0%})",
                similarity=similarity
            )
        else:
            return VerificationResult(
                status=VerificationStatus.SUSPICIOUS,
                reason=f"средняя_схожесть ({similarity:.1%})",
                similarity=similarity
            )

    def clean_for_comparison(self, text: str) -> str:
        """
        Очищает текст для сравнения.
        Убирает пробелы и пунктуацию, оставляя только буквы.
        """
        # Убираем всё кроме букв
        cleaned = re.sub(r'[^а-яёa-z]', '', text.lower())
        return cleaned

    def _calculate_length_ratio(self, s1: str, s2: str) -> float:
        """Вычисляет относительную разницу длин."""
        if not s1 and not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return abs(len(s1) - len(s2)) / max_len

    def _count_word_loss(self, author: str, audio: str) -> int:
        """
        Считает количество потерянных слов.
        """
        author_words = set(self._extract_words(author))
        audio_words = set(self._extract_words(audio))

        # Слова которые есть в оригинале, но нет в транскрипции
        lost = author_words - audio_words

        return len(lost)

    def _extract_words(self, text: str) -> List[str]:
        """Извлекает слова из текста."""
        return [w.lower() for w in re.findall(r'[а-яёa-z]+', text, re.IGNORECASE)]

    def _find_different_words(self, author: str, audio: str) -> List[Tuple[str, str]]:
        """
        v1.1: Находит слова, которые РЕАЛЬНО отличаются буквами.

        Возвращает список пар (слово_автора, слово_транскрипта) где слова
        стоят на одной позиции, но отличаются написанием.

        Это ключевая проверка: если слова отличаются — это НЕ технический шум.
        """
        author_words = self._extract_words(author)
        audio_words = self._extract_words(audio)

        different = []
        # Сравниваем слова на одинаковых позициях
        for w1, w2 in zip(author_words, audio_words):
            if w1 != w2:
                different.append((w1, w2))

        return different

    def get_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Вычисляет схожесть через расстояние Левенштейна.

        Returns:
            Значение от 0 до 1 (1 = идентичны)
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Используем rapidfuzz если доступен
        try:
            from rapidfuzz.distance import Levenshtein
            distance = Levenshtein.distance(s1, s2)
        except ImportError:
            # Fallback на встроенную реализацию
            distance = self._levenshtein_distance(s1, s2)

        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Встроенная реализация расстояния Левенштейна."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

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

    def is_word_transposition(self, author: str, audio: str) -> bool:
        """
        Проверяет, является ли различие перестановкой слов.

        Пример: "ты что" vs "что ты" = True
        """
        author_words = self._extract_words(author)
        audio_words = self._extract_words(audio)

        # Одинаковое количество слов
        if len(author_words) != len(audio_words):
            return False

        # Одинаковый набор слов (возможно в другом порядке)
        return sorted(author_words) == sorted(audio_words)


# =============================================================================
# SINGLETON И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

_window_verifier_instance = None

def get_window_verifier() -> WindowVerifier:
    """Возвращает глобальный экземпляр WindowVerifier."""
    global _window_verifier_instance
    if _window_verifier_instance is None:
        _window_verifier_instance = WindowVerifier()
    return _window_verifier_instance


def verify_segment(author: str, audio: str) -> VerificationResult:
    """Быстрая функция для верификации сегмента."""
    return get_window_verifier().verify_segment(author, audio)


def is_technical_noise(author: str, audio: str) -> bool:
    """
    Проверяет, является ли различие техническим шумом.

    Returns:
        True если можно скрыть (TECHNICAL_OK)
    """
    result = verify_segment(author, audio)
    return result.status == VerificationStatus.TECHNICAL_OK


def is_word_transposition(author: str, audio: str) -> bool:
    """Проверяет, является ли различие перестановкой слов."""
    return get_window_verifier().is_word_transposition(author, audio)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_window_verifier():
    """Тест WindowVerifier."""
    print('=' * 60)
    print('ТЕСТ: WindowVerifier')
    print('=' * 60)

    verifier = WindowVerifier()

    # Тест 1: Перестановка слов
    print('\n1. Перестановка слов ("ты что" vs "что ты"):')
    result = verifier.verify_segment("ты что", "что ты")
    print(f'   Статус: {result.status.value}')
    print(f'   Причина: {result.reason}')
    print(f'   Схожесть: {result.similarity:.2%}')
    print(f'   is_word_transposition: {verifier.is_word_transposition("ты что", "что ты")}')

    # Тест 2: Потеря слова
    print('\n2. Потеря слова ("Ни. За. Что." vs "Ничто"):')
    result = verifier.verify_segment("Ни. За. Что.", "Ничто")
    print(f'   Статус: {result.status.value}')
    print(f'   Причина: {result.reason}')
    print(f'   Details: {result.details}')

    # Тест 3: Высокая схожесть
    print('\n3. Высокая схожесть ("средоточие" vs "средоточия"):')
    result = verifier.verify_segment(
        "это было средоточие всей силы",
        "это было средоточия всей силы"
    )
    print(f'   Статус: {result.status.value}')
    print(f'   Причина: {result.reason}')
    print(f'   Схожесть: {result.similarity:.2%}')

    # Тест 4: Реальная ошибка
    print('\n4. Реальная ошибка ("получится" vs "получилось"):')
    result = verifier.verify_segment(
        "у нас точно получится это сделать",
        "у нас точно получилось это сделать"
    )
    print(f'   Статус: {result.status.value}')
    print(f'   Причина: {result.reason}')
    print(f'   Схожесть: {result.similarity:.2%}')

    # Тест 5: Короткий сегмент
    print('\n5. Короткий сегмент ("да" vs "и"):')
    result = verifier.verify_segment("да", "и")
    print(f'   Статус: {result.status.value}')
    print(f'   Причина: {result.reason}')

    # Тест 6: clean_for_comparison
    print('\n6. Очистка текста:')
    text = "Ни, за... что!"
    cleaned = verifier.clean_for_comparison(text)
    print(f'   Исходный: "{text}"')
    print(f'   Очищенный: "{cleaned}"')

    print()
    print('=' * 60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('=' * 60)


if __name__ == '__main__':
    test_window_verifier()
