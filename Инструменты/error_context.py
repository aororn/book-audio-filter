#!/usr/bin/env python3
"""
Error Context v1.0 — Расширенный контекст ошибок из всех источников

Модуль для создания и управления расширенным контекстом ошибок:
- Создание TXT файлов из DOCX оригиналов
- Сохранение нормализованных версий
- Создание alignment.json с сегментами выравнивания
- Связывание ошибок с позициями во всех файлах

Использование:
    python error_context.py prepare 01       # Подготовить файлы для главы 1
    python error_context.py prepare --all    # Подготовить все главы
    python error_context.py analyze 01       # Анализ ошибок главы 1

v1.0 (2026-01-31): Начальная версия
    - Конвертация DOCX → TXT
    - Создание нормализованных файлов
    - Структура ErrorContext dataclass
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-31'

import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# ПУТИ К ФАЙЛАМ
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
ORIGINALS_DIR = BASE_DIR / 'Оригинал' / 'Главы'
RESULTS_DIR = BASE_DIR / 'Результаты проверки'
TRANSCRIPTIONS_DIR = BASE_DIR / 'Транскрибации'


# =============================================================================
# КОНВЕРТАЦИЯ DOCX → TXT
# =============================================================================

def docx_to_txt(docx_path: Path) -> str:
    """
    Конвертирует DOCX в plain text.

    Args:
        docx_path: путь к DOCX файлу

    Returns:
        Текст документа
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Требуется python-docx: pip install python-docx")

    doc = Document(str(docx_path))
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    return '\n\n'.join(paragraphs)


def normalize_text_for_comparison(text: str) -> str:
    """
    Нормализует текст для сравнения:
    - Приводит к нижнему регистру
    - Заменяет ё → е
    - Убирает пунктуацию (кроме дефисов в составных словах)
    - Нормализует пробелы

    Args:
        text: исходный текст

    Returns:
        Нормализованный текст
    """
    # Нижний регистр
    text = text.lower()

    # ё → е
    text = text.replace('ё', 'е')

    # Убираем пунктуацию, оставляя дефисы внутри слов
    # Сначала защищаем дефисы между буквами
    text = re.sub(r'(\w)-(\w)', r'\1HYPHEN\2', text)

    # Убираем всю пунктуацию
    text = re.sub(r'[^\w\s]', ' ', text)

    # Восстанавливаем дефисы
    text = text.replace('HYPHEN', '-')

    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_words(text: str) -> List[str]:
    """
    Разбивает текст на слова.

    Args:
        text: текст (может быть нормализованным или нет)

    Returns:
        Список слов
    """
    # Если текст ещё не нормализован — нормализуем для разбиения
    normalized = normalize_text_for_comparison(text)
    return normalized.split()


# =============================================================================
# СТРУКТУРЫ ДАННЫХ
# =============================================================================

@dataclass
class WordPosition:
    """Позиция слова в файле"""
    word_idx: int           # Индекс слова в массиве слов
    char_start: int         # Начальная позиция символа в тексте
    char_end: int           # Конечная позиция символа
    word: str               # Само слово
    word_normalized: str    # Нормализованная форма


@dataclass
class AlignmentSegment:
    """Сегмент выравнивания между оригиналом и транскрипцией"""
    segment_id: int
    anchor_before: Optional[str]    # Якорь до сегмента
    anchor_after: Optional[str]     # Якорь после сегмента

    # Временные границы (из транскрипции)
    time_start: float
    time_end: float

    # Границы в оригинале (индексы слов)
    original_start: int
    original_end: int

    # Границы в транскрипции (индексы слов)
    transcript_start: int
    transcript_end: int

    # ID ошибок в этом сегменте
    error_ids: List[str] = field(default_factory=list)


@dataclass
class ErrorLink:
    """Связь между ошибками (для merge/split артефактов)"""
    link_id: str
    error_ids: List[str]            # ID связанных ошибок
    link_type: str                  # merge_artifact, split_artifact
    pattern: str                    # "на+встречу=навстречу"
    original_parts: List[str]       # ["на", "встречу"]
    merged_form: Optional[str]      # "навстречу"
    confidence: float = 1.0


@dataclass
class ErrorContext:
    """Расширенный контекст ошибки из всех источников"""

    # ИДЕНТИФИКАЦИЯ
    error_id: str                   # UUID ошибки
    chapter: int
    error_type: str                 # substitution, insertion, deletion

    # СЛОВА
    wrong: str                      # Что распознал Яндекс / сказал чтец
    correct: str                    # Что в оригинале

    # ВРЕМЕННЫЕ ОКНА
    time: float                     # Точное время ошибки
    time_end: float                 # Конец слова
    window_start: float             # Начало окна выравнивания (сегмент)
    window_end: float               # Конец окна выравнивания
    context_start: float            # Начало расширенного контекста (±N сек)
    context_end: float              # Конец расширенного контекста

    # ПОЗИЦИИ В ФАЙЛАХ
    pos_transcript: int             # Индекс в транскрипции (массив слов)
    pos_transcript_char: int        # Позиция символа в транскрипции TXT
    pos_normalized: int             # Индекс в нормализованном файле
    pos_original: int               # Индекс в оригинале (массив слов)
    pos_original_char: int          # Позиция символа в оригинале TXT

    # КОНТЕКСТЫ (±N слов)
    context_transcript: List[str]   # Слова из транскрипции
    context_normalized: List[str]   # Слова из нормализованного
    context_original: List[str]     # Слова из оригинала
    context_aligned: str            # Весь сегмент выравнивания

    # СВЯЗАННЫЕ ОШИБКИ (для слияния/разбиения)
    linked_error_ids: List[str] = field(default_factory=list)
    link_type: Optional[str] = None   # merge_artifact, split_artifact, None
    merged_form: Optional[str] = None # "навстречу" (если это split)
    split_parts: List[str] = field(default_factory=list)  # ["на", "встречу"]

    # МЕТАДАННЫЕ
    segment_id: int = -1            # ID сегмента выравнивания
    is_boundary: bool = False       # На границе сегмента?

    # МОРФОЛОГИЯ И СЕМАНТИКА (из существующих полей)
    lemma_wrong: Optional[str] = None
    lemma_correct: Optional[str] = None
    pos_wrong: Optional[str] = None
    pos_correct: Optional[str] = None
    same_lemma: bool = False
    same_pos: bool = False
    semantic_similarity: float = 0.0
    phonetic_similarity: float = 0.0

    # ФИЛЬТРАЦИЯ
    is_filtered: bool = False
    filter_reason: Optional[str] = None
    is_golden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для JSON"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorContext':
        """Создаёт из словаря"""
        return cls(**data)


# =============================================================================
# ПОДГОТОВКА ФАЙЛОВ
# =============================================================================

def get_chapter_paths(chapter: int) -> Dict[str, Path]:
    """
    Возвращает пути ко всем файлам главы.

    Args:
        chapter: номер главы (1-5)

    Returns:
        Словарь с путями к файлам
    """
    chapter_str = f"{chapter:02d}"
    results_chapter_dir = RESULTS_DIR / chapter_str

    # Ищем оригинал (разные форматы именования)
    original_docx = None
    for pattern in [f'Глава {chapter}.docx', f'Глава{chapter}.docx']:
        path = ORIGINALS_DIR / pattern
        if path.exists():
            original_docx = path
            break

    # Для главы 5 может быть TXT
    original_txt_existing = ORIGINALS_DIR / f'Глава{chapter}.txt'

    return {
        'chapter': chapter,
        'chapter_str': chapter_str,

        # Директории
        'results_dir': results_chapter_dir,

        # Оригинал
        'original_docx': original_docx,
        'original_txt_existing': original_txt_existing if original_txt_existing.exists() else None,

        # Новые файлы (будут созданы)
        'original_txt': results_chapter_dir / f'{chapter_str}_original.txt',
        'original_normalized': results_chapter_dir / f'{chapter_str}_original_normalized.txt',
        'transcript_normalized': results_chapter_dir / f'{chapter_str}_transcript_normalized.txt',
        'alignment_json': results_chapter_dir / f'{chapter_str}_alignment.json',
        'context_json': results_chapter_dir / f'{chapter_str}_error_contexts.json',

        # Существующие файлы
        'compared_json': results_chapter_dir / f'{chapter_str}_compared.json',
        'filtered_json': results_chapter_dir / f'{chapter_str}_filtered.json',
        'transcript_json': results_chapter_dir / f'{chapter_str}_transcript.json',
    }


def prepare_chapter_files(chapter: int, force: bool = False) -> Dict[str, Any]:
    """
    Подготавливает все необходимые файлы для главы.

    Создаёт:
    - {chapter}_original.txt — оригинал как plain text
    - {chapter}_original_normalized.txt — нормализованный оригинал
    - {chapter}_transcript_normalized.txt — нормализованная транскрипция

    Args:
        chapter: номер главы
        force: перезаписать существующие файлы

    Returns:
        Статистика созданных файлов
    """
    paths = get_chapter_paths(chapter)
    stats = {
        'chapter': chapter,
        'files_created': [],
        'files_skipped': [],
        'errors': [],
    }

    # Создаём директорию если нет
    paths['results_dir'].mkdir(parents=True, exist_ok=True)

    # 1. Создаём TXT из оригинала
    original_text = None

    if paths['original_txt'].exists() and not force:
        stats['files_skipped'].append(str(paths['original_txt']))
        original_text = paths['original_txt'].read_text(encoding='utf-8')
    else:
        try:
            if paths['original_docx'] and paths['original_docx'].exists():
                original_text = docx_to_txt(paths['original_docx'])
                paths['original_txt'].write_text(original_text, encoding='utf-8')
                stats['files_created'].append(str(paths['original_txt']))
            elif paths['original_txt_existing']:
                # Копируем существующий TXT
                original_text = paths['original_txt_existing'].read_text(encoding='utf-8')
                paths['original_txt'].write_text(original_text, encoding='utf-8')
                stats['files_created'].append(str(paths['original_txt']))
            else:
                stats['errors'].append(f"Не найден оригинал для главы {chapter}")
        except Exception as e:
            stats['errors'].append(f"Ошибка конвертации DOCX: {e}")

    # 2. Создаём нормализованный оригинал
    if original_text:
        if paths['original_normalized'].exists() and not force:
            stats['files_skipped'].append(str(paths['original_normalized']))
        else:
            normalized_original = normalize_text_for_comparison(original_text)
            paths['original_normalized'].write_text(normalized_original, encoding='utf-8')
            stats['files_created'].append(str(paths['original_normalized']))

    # 3. Создаём нормализованную транскрипцию из JSON
    if paths['transcript_json'].exists():
        if paths['transcript_normalized'].exists() and not force:
            stats['files_skipped'].append(str(paths['transcript_normalized']))
        else:
            try:
                transcript_text = extract_text_from_transcript(paths['transcript_json'])
                normalized_transcript = normalize_text_for_comparison(transcript_text)
                paths['transcript_normalized'].write_text(normalized_transcript, encoding='utf-8')
                stats['files_created'].append(str(paths['transcript_normalized']))
            except Exception as e:
                stats['errors'].append(f"Ошибка обработки транскрипции: {e}")
    else:
        stats['errors'].append(f"Не найден transcript.json для главы {chapter}")

    return stats


def extract_text_from_transcript(transcript_path: Path) -> str:
    """
    Извлекает текст из JSON транскрипции Яндекса.

    Args:
        transcript_path: путь к JSON файлу

    Returns:
        Полный текст транскрипции
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words = []

    # Формат Яндекс SpeechKit
    if 'chunks' in data:
        for chunk in data['chunks']:
            alternatives = chunk.get('alternatives', [])
            if alternatives:
                for word_data in alternatives[0].get('words', []):
                    words.append(word_data.get('word', ''))

    return ' '.join(words)


def extract_words_with_timing(transcript_path: Path) -> List[Dict[str, Any]]:
    """
    Извлекает слова с таймингами из транскрипции.

    Args:
        transcript_path: путь к JSON файлу

    Returns:
        Список словарей {word, start_time, end_time, confidence}
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words = []

    if 'chunks' in data:
        for chunk in data['chunks']:
            alternatives = chunk.get('alternatives', [])
            if alternatives:
                for word_data in alternatives[0].get('words', []):
                    # Парсим время (формат "3.139s")
                    start_str = word_data.get('startTime', '0s')
                    end_str = word_data.get('endTime', '0s')

                    start_time = float(start_str.rstrip('s')) if start_str else 0.0
                    end_time = float(end_str.rstrip('s')) if end_str else 0.0

                    words.append({
                        'word': word_data.get('word', ''),
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': word_data.get('confidence', 1.0),
                    })

    return words


# =============================================================================
# ПОСТРОЕНИЕ ПОЗИЦИОННОГО ИНДЕКСА
# =============================================================================

def build_word_positions(text: str) -> List[WordPosition]:
    """
    Строит индекс позиций всех слов в тексте.

    Args:
        text: исходный текст

    Returns:
        Список WordPosition для каждого слова
    """
    positions = []
    normalized_text = normalize_text_for_comparison(text)
    words = normalized_text.split()

    # Находим позиции в нормализованном тексте
    current_pos = 0
    for idx, word in enumerate(words):
        start = normalized_text.find(word, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(word)

        positions.append(WordPosition(
            word_idx=idx,
            char_start=start,
            char_end=end,
            word=word,
            word_normalized=word.lower().replace('ё', 'е'),
        ))

        current_pos = end

    return positions


def find_word_position(
    word: str,
    positions: List[WordPosition],
    near_idx: int = -1,
    max_distance: int = 50
) -> Optional[int]:
    """
    Находит позицию слова в списке позиций.

    Args:
        word: искомое слово
        positions: список WordPosition
        near_idx: предпочтительная позиция (для поиска рядом)
        max_distance: максимальное расстояние от near_idx

    Returns:
        Индекс слова или None
    """
    word_norm = word.lower().replace('ё', 'е')

    # Если есть near_idx — ищем сначала рядом
    if near_idx >= 0:
        start = max(0, near_idx - max_distance)
        end = min(len(positions), near_idx + max_distance)

        for i in range(start, end):
            if positions[i].word_normalized == word_norm:
                return i

    # Иначе — полный поиск
    for i, pos in enumerate(positions):
        if pos.word_normalized == word_norm:
            return i

    return None


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def generate_error_id() -> str:
    """Генерирует уникальный ID для ошибки"""
    return str(uuid.uuid4())[:8]


def get_context_window(
    words: List[str],
    center_idx: int,
    window_size: int = 10
) -> List[str]:
    """
    Извлекает окно контекста вокруг слова.

    Args:
        words: список слов
        center_idx: индекс центрального слова
        window_size: количество слов с каждой стороны

    Returns:
        Список слов контекста
    """
    start = max(0, center_idx - window_size)
    end = min(len(words), center_idx + window_size + 1)
    return words[start:end]


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Error Context — Расширенный контекст ошибок'
    )

    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # Команда prepare
    prep_parser = subparsers.add_parser('prepare', help='Подготовить файлы')
    prep_parser.add_argument(
        'chapter',
        nargs='?',
        help='Номер главы (1-5) или --all для всех'
    )
    prep_parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Обработать все главы'
    )
    prep_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Перезаписать существующие файлы'
    )

    # Команда info
    info_parser = subparsers.add_parser('info', help='Информация о главе')
    info_parser.add_argument('chapter', type=int, help='Номер главы')

    args = parser.parse_args()

    print(f"Error Context v{VERSION}")
    print("=" * 60)

    if args.command == 'prepare':
        if args.all:
            chapters = [1, 2, 3, 4, 5]
        elif args.chapter:
            chapters = [int(args.chapter)]
        else:
            print("Укажите номер главы или --all")
            return

        for chapter in chapters:
            print(f"\n📖 Глава {chapter}:")
            stats = prepare_chapter_files(chapter, force=args.force)

            if stats['files_created']:
                print(f"  ✅ Создано: {len(stats['files_created'])} файлов")
                for f in stats['files_created']:
                    print(f"     - {Path(f).name}")

            if stats['files_skipped']:
                print(f"  ⏭️  Пропущено: {len(stats['files_skipped'])} файлов")

            if stats['errors']:
                print(f"  ❌ Ошибки:")
                for e in stats['errors']:
                    print(f"     - {e}")

    elif args.command == 'info':
        paths = get_chapter_paths(args.chapter)
        print(f"\n📖 Глава {args.chapter}:")

        for name, path in paths.items():
            if isinstance(path, Path):
                status = "✅" if path.exists() else "❌"
                print(f"  {status} {name}: {path.name if path else 'N/A'}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
