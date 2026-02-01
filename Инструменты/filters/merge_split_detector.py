#!/usr/bin/env python3
"""
Merge/Split Detector v1.0 — Детектор артефактов слияния и разбиения слов

Детектирует паттерны, когда Яндекс сливает или разбивает слова:

СЛИЯНИЕ (Яндекс слил слова оригинала):
  Оригинал:    "на встречу мучали"  (3 слова)
  Транскрипт:  "навстречу мучили"   (2 слова)
  Ошибки:      substitution "на"→"навстречу" + deletion "встречу"

РАЗБИЕНИЕ (Яндекс разбил слово оригинала):
  Оригинал:    "яше"         (1 слово)
  Транскрипт:  "я ше"        (2 слова)
  Ошибки:      substitution "яше"→"я" + insertion "ше"

Использование:
    from filters.merge_split_detector import detect_merge_patterns, detect_split_patterns

v1.0 (2026-01-31): Начальная версия
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-31'

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set


# =============================================================================
# СТРУКТУРЫ ДАННЫХ
# =============================================================================

@dataclass
class MergeSplitPattern:
    """Паттерн слияния или разбиения"""
    pattern_id: str                         # UUID паттерна
    pattern_type: str                       # 'merge' или 'split'

    # Участвующие ошибки
    error_ids: List[str]                    # ID ошибок в паттерне
    error_indices: List[int]                # Индексы в списке ошибок

    # Детали паттерна
    original_parts: List[str]               # Слова из оригинала
    transcript_parts: List[str]             # Слова из транскрипции
    merged_form: str                        # Слитая форма
    pattern_str: str                        # Человекочитаемый паттерн

    # Времення
    time_start: float
    time_end: float

    # Типы ошибок
    error_types: List[str]                  # ['substitution', 'deletion']

    # Уверенность
    confidence: float = 1.0

    # Дополнительно
    chapter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# ИЗВЕСТНЫЕ ПАТТЕРНЫ СЛИЯНИЯ
# =============================================================================

# Слова, которые Яндекс часто сливает с предыдущим/следующим
COMMON_MERGE_PREFIXES = {
    'на', 'во', 'по', 'за', 'от', 'до', 'из', 'со', 'об', 'под',
    'над', 'пред', 'при', 'про', 'пере', 'вы', 'вс', 'вз', 'раз',
    'рас', 'без', 'бес', 'не', 'ни', 'как', 'так', 'что', 'кто',
}

COMMON_MERGE_SUFFIXES = {
    'то', 'либо', 'нибудь', 'ка', 'таки', 'же', 'ли', 'бы',
    'встречу', 'время', 'место', 'верх', 'низ', 'перёд', 'зад',
}

# Известные слитные формы
KNOWN_MERGED_FORMS = {
    'навстречу': ('на', 'встречу'),
    'вовремя': ('во', 'время'),
    'наверх': ('на', 'верх'),
    'также': ('так', 'же'),
    'тоже': ('то', 'же'),
    'чтобы': ('что', 'бы'),
    'кто-то': ('кто', 'то'),
    'что-то': ('что', 'то'),
    'как-то': ('как', 'то'),
    'где-то': ('где', 'то'),
    'когда-то': ('когда', 'то'),
    'почему-то': ('почему', 'то'),
    'откуда-то': ('откуда', 'то'),
    'яша': ('я', 'ша'),
    'яже': ('я', 'же'),
    'заян': ('за', 'ян'),
}


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def normalize_word(word: str) -> str:
    """Нормализует слово для сравнения"""
    return word.lower().replace('ё', 'е').replace('-', '').strip()


def generate_pattern_id() -> str:
    """Генерирует уникальный ID паттерна"""
    return str(uuid.uuid4())[:8]


def get_error_word(error: Dict[str, Any], which: str = 'transcript') -> str:
    """
    Извлекает слово из ошибки.

    Args:
        error: словарь ошибки
        which: 'transcript' или 'original'
    """
    if which == 'transcript':
        return error.get('transcript', '') or error.get('wrong', '')
    else:
        return error.get('original', '') or error.get('correct', '')


def get_error_time(error: Dict[str, Any]) -> float:
    """Извлекает время ошибки"""
    return error.get('time', error.get('time_seconds', 0))


def get_error_type(error: Dict[str, Any]) -> str:
    """Извлекает тип ошибки"""
    return error.get('type', error.get('error_type', ''))


def get_error_id(error: Dict[str, Any], index: int = -1) -> str:
    """Извлекает или генерирует ID ошибки"""
    eid = error.get('error_id', '')
    if not eid:
        eid = f"err_{index}" if index >= 0 else generate_pattern_id()
    return eid


# =============================================================================
# ДЕТЕКТОРЫ ПАТТЕРНОВ
# =============================================================================

def is_merge_candidate(err1: Dict, err2: Dict, time_threshold: float = 2.0) -> bool:
    """
    Проверяет, являются ли две ошибки кандидатами на merge паттерн.

    Merge паттерн:
    - err1: substitution orig1 → merged_word
    - err2: deletion orig2
    - merged_word == orig1 + orig2
    """
    type1 = get_error_type(err1)
    type2 = get_error_type(err2)

    # Проверяем типы
    if not ((type1 == 'substitution' and type2 == 'deletion') or
            (type1 == 'deletion' and type2 == 'substitution')):
        return False

    # Проверяем время
    time1 = get_error_time(err1)
    time2 = get_error_time(err2)
    if abs(time1 - time2) > time_threshold:
        return False

    return True


def check_merge_pattern(err1: Dict, err2: Dict) -> Optional[MergeSplitPattern]:
    """
    Проверяет, образуют ли две ошибки merge паттерн.

    Паттерн слияния:
    - Яндекс распознал "навстречу" вместо "на встречу"
    - Создаёт: substitution "на"→"навстречу" + deletion "встречу"
    """
    type1 = get_error_type(err1)
    type2 = get_error_type(err2)

    # Определяем какая ошибка substitution, какая deletion
    if type1 == 'substitution' and type2 == 'deletion':
        sub_err, del_err = err1, err2
    elif type1 == 'deletion' and type2 == 'substitution':
        sub_err, del_err = err2, err1
    else:
        return None

    # Слова
    merged = normalize_word(get_error_word(sub_err, 'transcript'))
    part1 = normalize_word(get_error_word(sub_err, 'original'))
    part2 = normalize_word(get_error_word(del_err, 'original'))

    if not merged or not part1 or not part2:
        return None

    # Проверяем: merged == part1 + part2
    combined = part1 + part2
    combined_no_hyphen = combined.replace('-', '')

    if merged == combined or merged == combined_no_hyphen:
        return MergeSplitPattern(
            pattern_id=generate_pattern_id(),
            pattern_type='merge',
            error_ids=[get_error_id(err1), get_error_id(err2)],
            error_indices=[],  # Заполняется позже
            original_parts=[part1, part2],
            transcript_parts=[merged],
            merged_form=merged,
            pattern_str=f"{part1}+{part2}={merged}",
            time_start=min(get_error_time(err1), get_error_time(err2)),
            time_end=max(get_error_time(err1), get_error_time(err2)),
            error_types=[type1, type2],
            confidence=1.0,
        )

    # Проверяем известные формы
    if merged in KNOWN_MERGED_FORMS:
        known_parts = KNOWN_MERGED_FORMS[merged]
        if (part1, part2) == known_parts or (part2, part1) == known_parts:
            return MergeSplitPattern(
                pattern_id=generate_pattern_id(),
                pattern_type='merge',
                error_ids=[get_error_id(err1), get_error_id(err2)],
                error_indices=[],
                original_parts=list(known_parts),
                transcript_parts=[merged],
                merged_form=merged,
                pattern_str=f"{known_parts[0]}+{known_parts[1]}={merged}",
                time_start=min(get_error_time(err1), get_error_time(err2)),
                time_end=max(get_error_time(err1), get_error_time(err2)),
                error_types=[type1, type2],
                confidence=0.9,
            )

    return None


def is_split_candidate(err1: Dict, err2: Dict, time_threshold: float = 2.0) -> bool:
    """
    Проверяет, являются ли две ошибки кандидатами на split паттерн.

    Split паттерн:
    - err1: substitution orig → part1
    - err2: insertion part2
    - orig == part1 + part2
    """
    type1 = get_error_type(err1)
    type2 = get_error_type(err2)

    # Проверяем типы
    if not ((type1 == 'substitution' and type2 == 'insertion') or
            (type1 == 'insertion' and type2 == 'substitution')):
        return False

    # Проверяем время
    time1 = get_error_time(err1)
    time2 = get_error_time(err2)
    if abs(time1 - time2) > time_threshold:
        return False

    return True


def check_split_pattern(err1: Dict, err2: Dict) -> Optional[MergeSplitPattern]:
    """
    Проверяет, образуют ли две ошибки split паттерн.

    Паттерн разбиения:
    - Яндекс разбил "яша" на "я ша"
    - Создаёт: substitution "яша"→"я" + insertion "ша"
    """
    type1 = get_error_type(err1)
    type2 = get_error_type(err2)

    # Определяем какая ошибка substitution, какая insertion
    if type1 == 'substitution' and type2 == 'insertion':
        sub_err, ins_err = err1, err2
    elif type1 == 'insertion' and type2 == 'substitution':
        sub_err, ins_err = err2, err1
    else:
        return None

    # Слова
    original = normalize_word(get_error_word(sub_err, 'original'))
    part1 = normalize_word(get_error_word(sub_err, 'transcript'))
    part2 = normalize_word(get_error_word(ins_err, 'transcript'))

    if not original or not part1 or not part2:
        return None

    # Проверяем: original == part1 + part2
    combined = part1 + part2
    combined_no_hyphen = combined.replace('-', '')

    if original == combined or original == combined_no_hyphen:
        return MergeSplitPattern(
            pattern_id=generate_pattern_id(),
            pattern_type='split',
            error_ids=[get_error_id(err1), get_error_id(err2)],
            error_indices=[],
            original_parts=[original],
            transcript_parts=[part1, part2],
            merged_form=original,
            pattern_str=f"{original}={part1}+{part2}",
            time_start=min(get_error_time(err1), get_error_time(err2)),
            time_end=max(get_error_time(err1), get_error_time(err2)),
            error_types=[type1, type2],
            confidence=1.0,
        )

    # Проверяем обратный порядок частей
    combined_rev = part2 + part1
    if original == combined_rev:
        return MergeSplitPattern(
            pattern_id=generate_pattern_id(),
            pattern_type='split',
            error_ids=[get_error_id(err1), get_error_id(err2)],
            error_indices=[],
            original_parts=[original],
            transcript_parts=[part2, part1],
            merged_form=original,
            pattern_str=f"{original}={part2}+{part1}",
            time_start=min(get_error_time(err1), get_error_time(err2)),
            time_end=max(get_error_time(err1), get_error_time(err2)),
            error_types=[type1, type2],
            confidence=0.95,
        )

    return None


# =============================================================================
# ГЛАВНЫЕ ФУНКЦИИ ДЕТЕКЦИИ
# =============================================================================

def detect_merge_patterns(
    errors: List[Dict[str, Any]],
    time_threshold: float = 2.0,
    chapter: int = 0
) -> List[MergeSplitPattern]:
    """
    Детектирует паттерны слияния в списке ошибок.

    Args:
        errors: список ошибок из compared.json
        time_threshold: максимальная разница во времени между ошибками
        chapter: номер главы

    Returns:
        Список найденных паттернов слияния
    """
    patterns = []
    used_indices: Set[int] = set()

    # Сортируем по времени
    sorted_errors = sorted(enumerate(errors), key=lambda x: get_error_time(x[1]))

    for i, (idx1, err1) in enumerate(sorted_errors):
        if idx1 in used_indices:
            continue

        for idx2, err2 in sorted_errors[i+1:]:
            if idx2 in used_indices:
                continue

            # Проверяем расстояние по времени
            if get_error_time(err2) - get_error_time(err1) > time_threshold:
                break  # Дальше ещё больше — выходим

            if is_merge_candidate(err1, err2, time_threshold):
                pattern = check_merge_pattern(err1, err2)
                if pattern:
                    pattern.chapter = chapter
                    pattern.error_indices = [idx1, idx2]
                    patterns.append(pattern)
                    used_indices.add(idx1)
                    used_indices.add(idx2)
                    break  # Нашли пару для err1, переходим к следующей

    return patterns


def detect_split_patterns(
    errors: List[Dict[str, Any]],
    time_threshold: float = 2.0,
    chapter: int = 0
) -> List[MergeSplitPattern]:
    """
    Детектирует паттерны разбиения в списке ошибок.

    Args:
        errors: список ошибок из compared.json
        time_threshold: максимальная разница во времени между ошибками
        chapter: номер главы

    Returns:
        Список найденных паттернов разбиения
    """
    patterns = []
    used_indices: Set[int] = set()

    # Сортируем по времени
    sorted_errors = sorted(enumerate(errors), key=lambda x: get_error_time(x[1]))

    for i, (idx1, err1) in enumerate(sorted_errors):
        if idx1 in used_indices:
            continue

        for idx2, err2 in sorted_errors[i+1:]:
            if idx2 in used_indices:
                continue

            if get_error_time(err2) - get_error_time(err1) > time_threshold:
                break

            if is_split_candidate(err1, err2, time_threshold):
                pattern = check_split_pattern(err1, err2)
                if pattern:
                    pattern.chapter = chapter
                    pattern.error_indices = [idx1, idx2]
                    patterns.append(pattern)
                    used_indices.add(idx1)
                    used_indices.add(idx2)
                    break

    return patterns


def detect_all_patterns(
    errors: List[Dict[str, Any]],
    time_threshold: float = 2.0,
    chapter: int = 0
) -> Tuple[List[MergeSplitPattern], List[MergeSplitPattern]]:
    """
    Детектирует все паттерны слияния и разбиения.

    Returns:
        (merge_patterns, split_patterns)
    """
    merge_patterns = detect_merge_patterns(errors, time_threshold, chapter)
    split_patterns = detect_split_patterns(errors, time_threshold, chapter)
    return merge_patterns, split_patterns


def get_linked_error_indices(patterns: List[MergeSplitPattern]) -> Set[int]:
    """
    Возвращает индексы всех ошибок, участвующих в паттернах.
    """
    indices = set()
    for p in patterns:
        indices.update(p.error_indices)
    return indices


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_detector():
    """Тестирует детектор на примерах"""
    print(f"Merge/Split Detector v{VERSION}")
    print("=" * 60)

    # Тестовые примеры
    test_errors = [
        # Merge: "на встречу" → "навстречу"
        {'type': 'substitution', 'original': 'на', 'transcript': 'навстречу', 'time': 10.0},
        {'type': 'deletion', 'original': 'встречу', 'transcript': '', 'time': 10.5},

        # Split: "яша" → "я ша"
        {'type': 'substitution', 'original': 'яша', 'transcript': 'я', 'time': 20.0},
        {'type': 'insertion', 'original': '', 'transcript': 'ша', 'time': 20.3},

        # Обычная ошибка (не паттерн)
        {'type': 'substitution', 'original': 'мучали', 'transcript': 'мучили', 'time': 30.0},

        # Merge: "во время" → "вовремя"
        {'type': 'substitution', 'original': 'во', 'transcript': 'вовремя', 'time': 40.0},
        {'type': 'deletion', 'original': 'время', 'transcript': '', 'time': 40.5},
    ]

    merge_patterns, split_patterns = detect_all_patterns(test_errors, chapter=1)

    print(f"\nMerge patterns ({len(merge_patterns)}):")
    for p in merge_patterns:
        print(f"  - {p.pattern_str} (indices: {p.error_indices})")

    print(f"\nSplit patterns ({len(split_patterns)}):")
    for p in split_patterns:
        print(f"  - {p.pattern_str} (indices: {p.error_indices})")

    linked = get_linked_error_indices(merge_patterns + split_patterns)
    print(f"\nLinked error indices: {sorted(linked)}")

    # Проверяем что обычная ошибка не попала
    assert 4 not in linked, "Обычная ошибка не должна быть в паттернах"
    print("\n[OK] Тесты пройдены!")


if __name__ == '__main__':
    test_detector()
