#!/usr/bin/env python3
"""
batch_process_transcripts.py v1.0

Пакетная обработка транскрипций через пайплайн сравнения и фильтрации.
Обрабатывает все транскрипции с суффиксом _NEW и сохраняет результаты.

Использование:
    python batch_process_transcripts.py                    # Все NEW транскрипции
    python batch_process_transcripts.py --filter 01       # Только глава 1
    python batch_process_transcripts.py --list            # Показать файлы
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial

# Unbuffered print
print = partial(print, flush=True)

# Добавляем путь к инструментам
sys.path.insert(0, str(Path(__file__).parent))

# Импортируем модули проекта
from config import PROJECT_DIR, TRANSCRIPTIONS_DIR, RESULTS_DIR, ORIGINAL_DIR
from smart_compare import smart_compare
from text_normalizer import normalize_file
from golden_filter import filter_errors


def find_new_transcripts(filter_pattern=None):
    """Находит все NEW транскрипции."""
    transcripts = []

    for path in TRANSCRIPTIONS_DIR.rglob('*_NEW_*.json'):
        if filter_pattern and filter_pattern not in path.name:
            continue
        transcripts.append(path)

    return sorted(transcripts)


def get_chapter_from_path(path):
    """Извлекает номер главы из пути."""
    # Транскрибации/Глава1/01_xxx.json -> 01
    name = path.name
    if name.startswith('01'):
        return '01'
    elif name.startswith('02'):
        return '02'
    elif name.startswith('03'):
        return '03'
    return None


def get_original_path(chapter):
    """Получает путь к оригиналу главы."""
    originals_dir = ORIGINAL_DIR / 'Главы'

    # Убираем ведущий ноль для номера главы
    chapter_num = chapter.lstrip('0') or '0'

    # Варианты имён
    patterns = [
        f'Глава {chapter_num}.docx',
        f'Глава{chapter_num}.docx',
        f'глава {chapter_num}.docx',
        f'глава{chapter_num}.docx',
        f'Глава {chapter}.docx',
        f'Глава{chapter}.docx',
    ]

    for pattern in patterns:
        path = originals_dir / pattern
        if path.exists():
            return path

    # Поиск по номеру (без временных файлов ~$)
    for f in originals_dir.glob('*.docx'):
        if f.name.startswith('~$'):
            continue
        if chapter_num in f.name or chapter in f.name:
            return f

    return None


def process_transcript(transcript_path, output_dir=None):
    """
    Обрабатывает одну транскрипцию через пайплайн.

    Returns:
        dict с результатами или None при ошибке
    """
    chapter = get_chapter_from_path(transcript_path)
    if not chapter:
        print(f"  ✗ Не удалось определить главу: {transcript_path.name}")
        return None

    original_path = get_original_path(chapter)
    if not original_path:
        print(f"  ✗ Оригинал не найден для главы {chapter}")
        return None

    # Определяем выходную папку
    if output_dir is None:
        output_dir = RESULTS_DIR / chapter

    output_dir.mkdir(parents=True, exist_ok=True)

    # Имя для результатов
    base_name = transcript_path.stem  # 01_16kbps_transcript_NEW_20260125_1811

    print(f"\n{'='*60}")
    print(f"  Транскрипция: {transcript_path.name}")
    print(f"  Оригинал: {original_path.name}")
    print(f"  Глава: {chapter}")
    print(f"{'='*60}")

    try:
        # 1. Нормализация оригинала
        print("  [1/3] Нормализация оригинала...")
        normalized_path = output_dir / f'{base_name}_normalized.txt'
        normalize_file(str(original_path), str(normalized_path), force=True)

        # 2. Сравнение через smart_compare
        print("  [2/3] Сравнение текстов...")
        compared_path = output_dir / f'{base_name}_compared.json'

        # phantom_seconds=None включает автоопределение начала текста
        # Функция detect_phantom_seconds ищет первые слова оригинала в транскрипции
        result = smart_compare(
            str(transcript_path),
            str(normalized_path),
            output_path=str(compared_path),
            phantom_seconds=None,  # Автоопределение
            force=True
        )

        errors_data = result.get('errors', [])
        print(f"       Найдено несовпадений: {len(errors_data)}")

        # 3. Фильтрация
        print("  [3/3] Фильтрация ошибок...")
        filtered_errors, removed, filter_stats = filter_errors(errors_data)
        print(f"       После фильтрации: {len(filtered_errors)}")

        # Сохраняем filtered
        filtered_path = output_dir / f'{base_name}_filtered.json'
        with open(filtered_path, 'w', encoding='utf-8') as f:
            json.dump({
                'errors': filtered_errors,
                'stats': {
                    'total_before': len(errors_data),
                    'total_after': len(filtered_errors),
                    'removed': len(removed),
                    'filter_stats': filter_stats
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Сохранено: {filtered_path.name}")

        return {
            'transcript': transcript_path.name,
            'chapter': chapter,
            'total_before': len(errors_data),
            'total_after': len(filtered_errors),
            'filtered_path': str(filtered_path),
            'filter_stats': filter_stats
        }

    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Пакетная обработка транскрипций')
    parser.add_argument('--list', action='store_true', help='Только показать файлы')
    parser.add_argument('--filter', '-f', help='Фильтр по имени')
    args = parser.parse_args()

    transcripts = find_new_transcripts(args.filter)

    print(f"\nНайдено транскрипций: {len(transcripts)}")
    for t in transcripts:
        print(f"  {t.name}")

    if args.list:
        return

    if not transcripts:
        print("\nНет файлов для обработки")
        return

    print(f"\nНачинаю обработку {len(transcripts)} файлов...")

    results = []
    for transcript in transcripts:
        result = process_transcript(transcript)
        if result:
            results.append(result)

    # Итоги
    print(f"\n{'='*60}")
    print("ИТОГИ:")
    print(f"{'='*60}")

    total_errors = 0
    for r in results:
        print(f"\n{r['transcript']}:")
        print(f"  До фильтрации: {r['total_before']}")
        print(f"  После фильтрации: {r['total_after']}")
        total_errors += r['total_after']

    print(f"\nОбщее количество ошибок: {total_errors}")
    print(f"Успешно обработано: {len(results)}/{len(transcripts)}")

    # Сохраняем сводку
    summary_path = RESULTS_DIR / f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'processed': len(results),
            'total_errors': total_errors,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nСводка сохранена: {summary_path}")


if __name__ == '__main__':
    main()
