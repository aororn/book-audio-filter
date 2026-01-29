"""
AlignmentManager v1.1 — Управление выравниванием текста и транскрипции.

Макро-выравнивание через якоря:
1. Находим уникальные слова, встречающиеся в ОБОИХ текстах
2. Используем их как опорные точки для сегментации
3. Выравниваем каждый сегмент отдельно (точнее, чем весь текст сразу)

v1.1 (2026-01-29): Добавлены refine_large_segments() и check_density()
v1.0 (2026-01-29): Начальная версия
"""

VERSION = '1.2.0'
VERSION_DATE = '2026-01-29'

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter

# Импорт CharacterGuard
try:
    from filters.character_guard import (
        get_character_guard, is_character_name, is_anchor_candidate as cg_is_anchor_candidate
    )
    HAS_CHARACTER_GUARD = True
except ImportError:
    HAS_CHARACTER_GUARD = False
    is_character_name = lambda x: False
    cg_is_anchor_candidate = lambda x, min_len=6: len(x) >= min_len


# =============================================================================
# ТИПЫ ДАННЫХ
# =============================================================================

@dataclass
class AnchorPoint:
    """Якорная точка для синхронизации."""
    word: str              # Слово-якорь
    orig_idx: int          # Позиция в оригинале
    trans_idx: int         # Позиция в транскрипции
    confidence: float      # Уверенность (1.0 = уникальное совпадение)


@dataclass
class Segment:
    """Сегмент текста между якорями."""
    orig_start: int        # Начало в оригинале
    orig_end: int          # Конец в оригинале
    trans_start: int       # Начало в транскрипции
    trans_end: int         # Конец в транскрипции
    anchor_before: Optional[str] = None  # Якорь перед сегментом
    anchor_after: Optional[str] = None   # Якорь после сегмента


# =============================================================================
# ALIGNMENT MANAGER
# =============================================================================

class AlignmentManager:
    """
    Управляет выравниванием текста через систему якорей.

    Якорь — это слово, которое:
    1. Встречается РОВНО 1 раз в оригинале
    2. Встречается РОВНО 1 раз в транскрипции
    3. Длина >= min_length символов
    4. НЕ является именем персонажа (они повторяются)
    5. НЕ является частым термином
    """

    # Пороги для якорей
    DEFAULT_MIN_LENGTH = 6
    SUB_ANCHOR_MIN_LENGTH_LARGE_GAP = 4  # для gap > 150 слов
    SUB_ANCHOR_MIN_LENGTH_MEDIUM_GAP = 5  # для gap > 100 слов

    # Пороги для gap
    LARGE_GAP_THRESHOLD = 150
    MEDIUM_GAP_THRESHOLD = 100

    def __init__(self, min_length: int = None):
        """
        Args:
            min_length: минимальная длина слова для якоря
        """
        self.min_length = min_length or self.DEFAULT_MIN_LENGTH
        self._anchors: List[AnchorPoint] = []
        self._segments: List[Segment] = []

    def find_anchor_points(
        self,
        orig_words: List[str],
        trans_words: List[str],
        min_length: int = None
    ) -> List[AnchorPoint]:
        """
        Находит якорные точки между оригиналом и транскрипцией.

        Args:
            orig_words: нормализованные слова оригинала
            trans_words: нормализованные слова транскрипции
            min_length: минимальная длина (переопределяет self.min_length)

        Returns:
            Список AnchorPoint отсортированный по позиции в оригинале
        """
        min_len = min_length or self.min_length

        # Подсчитываем частоту слов
        orig_counts = Counter(orig_words)
        trans_counts = Counter(trans_words)

        # Находим уникальные слова в обоих текстах
        unique_in_orig = {w for w, c in orig_counts.items() if c == 1}
        unique_in_trans = {w for w, c in trans_counts.items() if c == 1}

        # Пересечение — кандидаты на якоря
        candidates = unique_in_orig & unique_in_trans

        # Фильтруем по критериям
        anchors = []
        for word in candidates:
            if not self._is_valid_anchor(word, min_len):
                continue

            # Находим позиции
            orig_idx = orig_words.index(word)
            trans_idx = trans_words.index(word)

            anchors.append(AnchorPoint(
                word=word,
                orig_idx=orig_idx,
                trans_idx=trans_idx,
                confidence=1.0
            ))

        # Сортируем по позиции в оригинале
        anchors.sort(key=lambda a: a.orig_idx)

        # v1.2: Фильтруем якоря по монотонности trans_idx
        # После сортировки по orig_idx, trans_idx тоже должен расти
        # Иначе якорь указывает на неправильное вхождение слова
        anchors = self._filter_monotonic(anchors)

        self._anchors = anchors
        return anchors

    def _filter_monotonic(self, anchors: List[AnchorPoint]) -> List[AnchorPoint]:
        """
        Фильтрует якоря, оставляя только монотонную последовательность по trans_idx.

        Использует алгоритм LIS (Longest Increasing Subsequence) для нахождения
        максимального подмножества якорей с монотонно возрастающим trans_idx.
        """
        if len(anchors) <= 1:
            return anchors

        # Алгоритм: жадный выбор монотонной последовательности
        # Проходим слева направо, отбрасывая якоря с trans_idx меньше предыдущего
        result = [anchors[0]]
        for anchor in anchors[1:]:
            if anchor.trans_idx > result[-1].trans_idx:
                result.append(anchor)
            # else: пропускаем якорь — он нарушает монотонность

        return result

    def _is_valid_anchor(self, word: str, min_length: int) -> bool:
        """Проверяет, подходит ли слово для якоря."""
        # Проверка длины
        if len(word) < min_length:
            return False

        # Проверка через CharacterGuard (если доступен)
        if HAS_CHARACTER_GUARD:
            return cg_is_anchor_candidate(word, min_length)

        # Fallback: просто проверяем длину и что не имя
        if is_character_name(word):
            return False

        return True

    def segment_by_anchors(
        self,
        orig_len: int,
        trans_len: int,
        anchors: List[AnchorPoint] = None
    ) -> List[Segment]:
        """
        Делит тексты на сегменты по якорям.

        Args:
            orig_len: длина оригинала (в словах)
            trans_len: длина транскрипции (в словах)
            anchors: список якорей (или использует self._anchors)

        Returns:
            Список сегментов
        """
        if anchors is None:
            anchors = self._anchors

        segments = []

        # Начальные границы
        prev_orig = 0
        prev_trans = 0
        prev_anchor = None

        for anchor in anchors:
            # Сегмент ДО текущего якоря
            if anchor.orig_idx > prev_orig or anchor.trans_idx > prev_trans:
                segments.append(Segment(
                    orig_start=prev_orig,
                    orig_end=anchor.orig_idx,
                    trans_start=prev_trans,
                    trans_end=anchor.trans_idx,
                    anchor_before=prev_anchor,
                    anchor_after=anchor.word
                ))

            prev_orig = anchor.orig_idx + 1
            prev_trans = anchor.trans_idx + 1
            prev_anchor = anchor.word

        # Последний сегмент (после последнего якоря)
        if prev_orig < orig_len or prev_trans < trans_len:
            segments.append(Segment(
                orig_start=prev_orig,
                orig_end=orig_len,
                trans_start=prev_trans,
                trans_end=trans_len,
                anchor_before=prev_anchor,
                anchor_after=None
            ))

        self._segments = segments
        return segments

    def get_dynamic_threshold(self, gap_words: int) -> int:
        """
        Возвращает минимальную длину якоря в зависимости от размера промежутка.

        Большие промежутки требуют более коротких якорей для точности.
        """
        if gap_words > self.LARGE_GAP_THRESHOLD:
            return self.SUB_ANCHOR_MIN_LENGTH_LARGE_GAP
        elif gap_words > self.MEDIUM_GAP_THRESHOLD:
            return self.SUB_ANCHOR_MIN_LENGTH_MEDIUM_GAP
        else:
            return self.DEFAULT_MIN_LENGTH

    def find_sub_anchors(
        self,
        orig_words: List[str],
        trans_words: List[str],
        segment: Segment
    ) -> List[AnchorPoint]:
        """
        Находит суб-якоря внутри большого сегмента.

        Используется для сегментов с gap > MEDIUM_GAP_THRESHOLD.
        """
        # Извлекаем слова сегмента
        seg_orig = orig_words[segment.orig_start:segment.orig_end]
        seg_trans = trans_words[segment.trans_start:segment.trans_end]

        # Определяем порог для суб-якорей
        gap = max(len(seg_orig), len(seg_trans))
        min_len = self.get_dynamic_threshold(gap)

        # Находим суб-якоря
        sub_anchors = self.find_anchor_points(seg_orig, seg_trans, min_len)

        # Корректируем позиции на глобальные
        for anchor in sub_anchors:
            anchor.orig_idx += segment.orig_start
            anchor.trans_idx += segment.trans_start

        return sub_anchors

    def get_stats(self) -> dict:
        """Возвращает статистику по якорям и сегментам."""
        return {
            'anchor_count': len(self._anchors),
            'segment_count': len(self._segments),
            'anchors': [a.word for a in self._anchors[:10]],  # первые 10
            'avg_segment_size': (
                sum(s.orig_end - s.orig_start for s in self._segments) / len(self._segments)
                if self._segments else 0
            )
        }

    def refine_large_segments(
        self,
        orig_words: List[str],
        trans_words: List[str],
        max_gap: int = 100
    ) -> Tuple[List['AnchorPoint'], List['Segment']]:
        """
        Разбивает большие сегменты суб-якорями.

        Для сегментов с gap > max_gap ищет дополнительные якоря
        с меньшей минимальной длиной (5 или 4 символа).

        Args:
            orig_words: нормализованные слова оригинала
            trans_words: нормализованные слова транскрипции
            max_gap: максимальный размер сегмента (по умолчанию 100 слов)

        Returns:
            (anchors, segments) — обновлённые списки
        """
        if not self._segments:
            return self._anchors, self._segments

        all_anchors = list(self._anchors)
        large_segments_found = 0

        for seg in self._segments:
            gap = seg.orig_end - seg.orig_start
            if gap > max_gap:
                large_segments_found += 1
                # Ищем суб-якоря
                sub_anchors = self.find_sub_anchors(orig_words, trans_words, seg)
                if sub_anchors:
                    all_anchors.extend(sub_anchors)

        if large_segments_found == 0:
            return self._anchors, self._segments

        # Сортируем все якоря по позиции
        all_anchors.sort(key=lambda a: a.orig_idx)

        # Удаляем дубликаты (по позиции)
        unique_anchors = []
        seen_positions = set()
        for anchor in all_anchors:
            pos_key = (anchor.orig_idx, anchor.trans_idx)
            if pos_key not in seen_positions:
                unique_anchors.append(anchor)
                seen_positions.add(pos_key)

        # Пересоздаём сегменты с новыми якорями
        self._anchors = unique_anchors
        refined_segments = self.segment_by_anchors(
            len(orig_words), len(trans_words), unique_anchors
        )

        return unique_anchors, refined_segments

    def check_density(self, segment: 'Segment', threshold: float = 0.15) -> bool:
        """
        Проверяет плотность сегмента.

        Возвращает True если разница в количестве слов > threshold.
        Это сигнал о возможных проблемах выравнивания.

        Args:
            segment: сегмент для проверки
            threshold: порог отклонения (по умолчанию 15%)

        Returns:
            True если сегмент требует внимания
        """
        orig_len = segment.orig_end - segment.orig_start
        trans_len = segment.trans_end - segment.trans_start

        if orig_len == 0 and trans_len == 0:
            return False

        max_len = max(orig_len, trans_len)
        if max_len == 0:
            return False

        density_diff = abs(orig_len - trans_len) / max_len
        return density_diff > threshold


# =============================================================================
# SINGLETON И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

_alignment_manager_instance = None

def get_alignment_manager() -> AlignmentManager:
    """Возвращает глобальный экземпляр AlignmentManager."""
    global _alignment_manager_instance
    if _alignment_manager_instance is None:
        _alignment_manager_instance = AlignmentManager()
    return _alignment_manager_instance


def find_anchors(orig_words: List[str], trans_words: List[str]) -> List[AnchorPoint]:
    """Быстрая функция для поиска якорей."""
    manager = get_alignment_manager()
    return manager.find_anchor_points(orig_words, trans_words)


def segment_texts(
    orig_words: List[str],
    trans_words: List[str]
) -> Tuple[List[AnchorPoint], List[Segment]]:
    """
    Находит якоря и делит тексты на сегменты.

    Returns:
        (anchors, segments)
    """
    manager = AlignmentManager()
    anchors = manager.find_anchor_points(orig_words, trans_words)
    segments = manager.segment_by_anchors(len(orig_words), len(trans_words))
    return anchors, segments


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_alignment_manager():
    """Тест AlignmentManager на примере."""
    print('=' * 60)
    print('ТЕСТ: AlignmentManager')
    print('=' * 60)

    # Пример текстов
    original = [
        'магистр', 'ордена', 'был', 'человеком', 'редкой', 'честности',
        'и', 'невероятной', 'силы', 'духа', 'который', 'проявлял',
        'в', 'моменты', 'величайшей', 'опасности', 'средоточие', 'его',
        'мыслей', 'было', 'направлено', 'на', 'защиту', 'павильона'
    ]

    transcript = [
        'магистр', 'ордена', 'был', 'человеком', 'редкой', 'честности',
        'и', 'невероятный', 'силы', 'духа', 'который', 'проявлял',
        'в', 'моменты', 'величайший', 'опасности', 'средоточие', 'его',
        'мыслей', 'было', 'направлена', 'на', 'защиту', 'павильона'
    ]

    print(f'Оригинал: {len(original)} слов')
    print(f'Транскрипция: {len(transcript)} слов')
    print()

    # Находим якоря
    manager = AlignmentManager()
    anchors = manager.find_anchor_points(original, transcript)

    print(f'Найдено якорей: {len(anchors)}')
    for a in anchors[:5]:
        print(f'  "{a.word}" @ orig[{a.orig_idx}] / trans[{a.trans_idx}]')
    if len(anchors) > 5:
        print(f'  ... и ещё {len(anchors) - 5}')
    print()

    # Делим на сегменты
    segments = manager.segment_by_anchors(len(original), len(transcript))

    print(f'Сегментов: {len(segments)}')
    for i, s in enumerate(segments[:3]):
        print(f'  Сегмент {i+1}: orig[{s.orig_start}:{s.orig_end}] / trans[{s.trans_start}:{s.trans_end}]')
        print(f'    между "{s.anchor_before}" и "{s.anchor_after}"')
    print()

    # Статистика
    stats = manager.get_stats()
    print(f'Статистика:')
    print(f'  Якорей: {stats["anchor_count"]}')
    print(f'  Сегментов: {stats["segment_count"]}')
    print(f'  Средний размер сегмента: {stats["avg_segment_size"]:.1f} слов')
    print()

    print('=' * 60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('=' * 60)


if __name__ == '__main__':
    test_alignment_manager()
