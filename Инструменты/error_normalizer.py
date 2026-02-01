#!/usr/bin/env python3
"""
Error Normalizer v1.0 — Единый источник правды для полей ошибок.

ПРОБЛЕМА:
В проекте используются разные названия полей для одних и тех же данных:
- JSON (compared/filtered): original, transcript
- БД (schema): correct, wrong
- Golden файлы: correct/original, wrong/transcript (дублирование!)

Это приводит к рассинхронизации БД и потере данных при сравнении.

РЕШЕНИЕ:
Единый модуль нормализации, который:
1. Принимает ошибку с ЛЮБЫМИ именами полей
2. Возвращает нормализованный словарь со СТАНДАРТНЫМИ именами
3. Используется во ВСЕХ местах работы с ошибками

СТАНДАРТ (соответствует схеме БД v2.2):
- original_word: слово из книги (correct в БД)
- transcript_word: слово из транскрипции/распознанное (wrong в БД)
- error_type: тип ошибки (substitution, insertion, deletion)
- time_seconds: время в секундах
- context: контекст ошибки

v1.0 (2026-02-01): Начальная версия
"""

VERSION = '1.0.0'

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# МАППИНГ ПОЛЕЙ
# =============================================================================

# Поля, которые означают "слово из книги" (original/correct)
ORIGINAL_FIELDS = ['original', 'correct', 'from_book', 'book_word']

# Поля, которые означают "слово из транскрипции" (transcript/wrong)
TRANSCRIPT_FIELDS = ['transcript', 'wrong', 'word', 'recognized', 'said_word']

# Поля времени
TIME_FIELDS = ['time', 'time_seconds', 'timecode']

# Поля типа ошибки
TYPE_FIELDS = ['type', 'error_type']

# Поля контекста
CONTEXT_FIELDS = ['context', 'context_original', 'book_context']


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# =============================================================================

def get_original_word(error: Dict[str, Any]) -> str:
    """
    Извлекает слово из книги (original/correct) из словаря ошибки.

    Приоритет полей:
    1. original (JSON формат)
    2. correct (БД/golden формат)
    3. from_book (устаревший формат)
    """
    for field in ORIGINAL_FIELDS:
        value = error.get(field)
        if value is not None:
            return str(value).lower() if value else ''
    return ''


def get_transcript_word(error: Dict[str, Any]) -> str:
    """
    Извлекает слово из транскрипции (transcript/wrong) из словаря ошибки.

    Приоритет полей:
    1. transcript (JSON формат)
    2. wrong (БД/golden формат)
    3. word (устаревший формат для insertion/deletion)
    """
    for field in TRANSCRIPT_FIELDS:
        value = error.get(field)
        if value is not None:
            return str(value).lower() if value else ''
    return ''


def get_time_seconds(error: Dict[str, Any]) -> float:
    """
    Извлекает время ошибки в секундах.

    Приоритет:
    1. time_seconds (числовой формат)
    2. time (может быть числом или строкой "MM:SS")
    """
    # Сначала пробуем time_seconds
    ts = error.get('time_seconds')
    if ts is not None:
        try:
            return float(ts)
        except (ValueError, TypeError):
            pass

    # Потом time
    t = error.get('time')
    if t is not None:
        if isinstance(t, (int, float)):
            return float(t)
        elif isinstance(t, str):
            # Пробуем парсить "MM:SS"
            try:
                if ':' in t:
                    parts = t.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                return float(t)
            except (ValueError, TypeError):
                pass

    return 0.0


def get_error_type(error: Dict[str, Any]) -> str:
    """Извлекает тип ошибки."""
    for field in TYPE_FIELDS:
        value = error.get(field)
        if value:
            return str(value)
    return 'substitution'


def get_context(error: Dict[str, Any]) -> str:
    """Извлекает контекст ошибки."""
    for field in CONTEXT_FIELDS:
        value = error.get(field)
        if value:
            return str(value)
    return ''


# =============================================================================
# НОРМАЛИЗАЦИЯ
# =============================================================================

@dataclass
class NormalizedError:
    """
    Нормализованная ошибка со стандартными полями.

    Использует консистентные имена:
    - original_word: слово из книги
    - transcript_word: слово из транскрипции
    """
    original_word: str      # Слово из книги (correct в БД)
    transcript_word: str    # Слово из транскрипции (wrong в БД)
    error_type: str
    time_seconds: float
    context: str = ''
    filter_reason: Optional[str] = None
    is_filtered: bool = False
    is_golden: bool = False
    error_id: str = ''

    # Дополнительные поля для совместимости
    phonetic_similarity: float = 0.0
    semantic_similarity: float = 0.0

    def to_db_dict(self) -> Dict[str, Any]:
        """
        Конвертирует в формат БД (wrong/correct).
        """
        return {
            'wrong': self.transcript_word,
            'correct': self.original_word,
            'error_type': self.error_type,
            'time_seconds': self.time_seconds,
            'context': self.context,
            'filter_reason': self.filter_reason,
            'is_filtered': 1 if self.is_filtered else 0,
            'is_golden': 1 if self.is_golden else 0,
            'error_id': self.error_id,
            'phonetic_similarity': self.phonetic_similarity,
            'semantic_similarity': self.semantic_similarity,
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Конвертирует в формат JSON (original/transcript).
        """
        return {
            'original': self.original_word,
            'transcript': self.transcript_word,
            'type': self.error_type,
            'time': self.time_seconds,
            'context': self.context,
            'filter_reason': self.filter_reason,
            'error_id': self.error_id,
            'phonetic_similarity': self.phonetic_similarity,
            'similarity': self.phonetic_similarity,
        }

    def make_key(self, time_precision: int = 1) -> str:
        """
        Создаёт уникальный ключ для сравнения ошибок.

        Args:
            time_precision: знаков после запятой для времени
        """
        time_key = round(self.time_seconds, time_precision)
        return f"{self.transcript_word}|{self.original_word}|{self.error_type}|{time_key}"


def normalize_error(error: Dict[str, Any]) -> NormalizedError:
    """
    Нормализует словарь ошибки в стандартный формат.

    Принимает ошибку с ЛЮБЫМИ именами полей:
    - JSON формат: original, transcript
    - БД формат: correct, wrong
    - Golden формат: correct/original, wrong/transcript
    - Устаревший: from_book, word

    Возвращает NormalizedError со стандартными полями.
    """
    return NormalizedError(
        original_word=get_original_word(error),
        transcript_word=get_transcript_word(error),
        error_type=get_error_type(error),
        time_seconds=get_time_seconds(error),
        context=get_context(error),
        filter_reason=error.get('filter_reason'),
        is_filtered=bool(error.get('is_filtered') or error.get('filter_reason')),
        is_golden=bool(error.get('is_golden')),
        error_id=error.get('error_id', ''),
        phonetic_similarity=error.get('phonetic_similarity', error.get('similarity', 0)),
        semantic_similarity=error.get('semantic_similarity', 0),
    )


def normalize_word(word: str) -> str:
    """
    Нормализует слово для сравнения.

    - Приводит к нижнему регистру
    - Заменяет ё на е
    - Удаляет пробелы по краям
    """
    if not word:
        return ''
    return word.lower().replace('ё', 'е').strip()


# =============================================================================
# СРАВНЕНИЕ ОШИБОК
# =============================================================================

def errors_match(
    error1: Dict[str, Any],
    error2: Dict[str, Any],
    time_tolerance: float = 15.0,
) -> bool:
    """
    Проверяет, совпадают ли две ошибки.

    Логика сравнения:
    1. Время должно быть в пределах tolerance
    2. Хотя бы одно слово должно совпадать (original или transcript)

    Args:
        error1, error2: Словари ошибок (любой формат)
        time_tolerance: Допустимая разница во времени (секунды)
    """
    norm1 = normalize_error(error1)
    norm2 = normalize_error(error2)

    # Проверка времени
    if abs(norm1.time_seconds - norm2.time_seconds) > time_tolerance:
        return False

    # Нормализуем слова
    orig1 = normalize_word(norm1.original_word)
    orig2 = normalize_word(norm2.original_word)
    trans1 = normalize_word(norm1.transcript_word)
    trans2 = normalize_word(norm2.transcript_word)

    # Проверка совпадения слов
    if orig1 and orig2 and orig1 == orig2:
        return True
    if trans1 and trans2 and trans1 == trans2:
        return True

    # Специальные случаи для insertion/deletion
    if orig1 == '' and orig2 == '' and trans1 and trans2 and trans1 == trans2:
        return True
    if trans1 == '' and trans2 == '' and orig1 and orig2 and orig1 == orig2:
        return True

    return False


def is_error_in_list(
    error: Dict[str, Any],
    error_list: list,
    time_tolerance: float = 15.0,
) -> bool:
    """
    Проверяет, есть ли ошибка в списке.

    Args:
        error: Словарь ошибки (любой формат)
        error_list: Список ошибок для поиска
        time_tolerance: Допустимая разница во времени
    """
    for e in error_list:
        if not isinstance(e, dict):
            continue
        if errors_match(error, e, time_tolerance):
            return True
    return False


def make_error_key(error: Dict[str, Any], time_precision: int = 1) -> str:
    """
    Создаёт уникальный ключ для ошибки.

    Ключ имеет формат: "transcript|original|type|time"

    Args:
        error: Словарь ошибки (любой формат)
        time_precision: знаков после запятой для времени
    """
    norm = normalize_error(error)
    return norm.make_key(time_precision)


# =============================================================================
# УТИЛИТЫ ДЛЯ МИГРАЦИИ
# =============================================================================

def convert_to_db_format(error: Dict[str, Any]) -> Dict[str, Any]:
    """
    Конвертирует ошибку в формат БД (wrong/correct).

    Сохраняет все оригинальные поля + добавляет стандартные.
    """
    norm = normalize_error(error)
    result = error.copy()

    # Добавляем/перезаписываем стандартные поля БД
    result['wrong'] = norm.transcript_word
    result['correct'] = norm.original_word
    result['error_type'] = norm.error_type
    result['time_seconds'] = norm.time_seconds

    return result


def convert_to_json_format(error: Dict[str, Any]) -> Dict[str, Any]:
    """
    Конвертирует ошибку в формат JSON (original/transcript).

    Сохраняет все оригинальные поля + добавляет стандартные.
    """
    norm = normalize_error(error)
    result = error.copy()

    # Добавляем/перезаписываем стандартные поля JSON
    result['original'] = norm.original_word
    result['transcript'] = norm.transcript_word
    result['type'] = norm.error_type
    result['time'] = norm.time_seconds

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print(f"Error Normalizer v{VERSION}")
    print("=" * 60)

    # Примеры использования
    print("\nПримеры нормализации:")

    # JSON формат
    json_error = {
        'original': 'живы',
        'transcript': 'живем',
        'type': 'substitution',
        'time': 59.83,
    }
    norm1 = normalize_error(json_error)
    print(f"\n1. JSON формат:")
    print(f"   Вход: original={json_error['original']}, transcript={json_error['transcript']}")
    print(f"   Выход: original_word={norm1.original_word}, transcript_word={norm1.transcript_word}")
    print(f"   Ключ: {norm1.make_key()}")

    # БД формат
    db_error = {
        'correct': 'живы',
        'wrong': 'живем',
        'error_type': 'substitution',
        'time_seconds': 59.83,
    }
    norm2 = normalize_error(db_error)
    print(f"\n2. БД формат:")
    print(f"   Вход: correct={db_error['correct']}, wrong={db_error['wrong']}")
    print(f"   Выход: original_word={norm2.original_word}, transcript_word={norm2.transcript_word}")
    print(f"   Ключ: {norm2.make_key()}")

    # Golden формат (дублирование)
    golden_error = {
        'correct': 'живы',
        'wrong': 'ЖИВЕМ',
        'original': 'живы',
        'transcript': 'ЖИВЕМ',
        'time_seconds': 59,
    }
    norm3 = normalize_error(golden_error)
    print(f"\n3. Golden формат:")
    print(f"   Вход: correct={golden_error['correct']}, original={golden_error['original']}")
    print(f"   Выход: original_word={norm3.original_word}, transcript_word={norm3.transcript_word}")
    print(f"   Ключ: {norm3.make_key()}")

    # Проверка совпадения
    print(f"\n4. Совпадение ошибок:")
    print(f"   JSON vs БД: {errors_match(json_error, db_error)}")
    print(f"   JSON vs Golden: {errors_match(json_error, golden_error)}")
    print(f"   БД vs Golden: {errors_match(db_error, golden_error)}")
