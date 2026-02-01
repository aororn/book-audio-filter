#!/usr/bin/env python3
"""
Turboscribe Merger v1.0
Склеивает CSV файлы транскрипций Турбоскрайб с коррекцией таймкодов.

Входные данные:
- {chapter}_yandex.csv — первые 30 минут
- {chapter}_окончание_48kbps.csv — остаток главы

Выходные данные:
- {chapter}_merged.csv — объединённый файл
- {chapter}_merged.txt — текст без таймкодов (для нормализации)
"""

import csv
import re
import sys
from pathlib import Path

VERSION = "1.0.0"

# Смещение для второй части: 30 минут = 1800 секунд = 1800000 мс
OFFSET_MS = 30 * 60 * 1000  # 1800000


def clean_text(text: str) -> str:
    """Убирает watermark TurboScribe и лишние пробелы."""
    # Убираем watermark
    text = re.sub(r'\(Transcribed by TurboScribe\.ai[^)]*\)', '', text)
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def read_csv(filepath: Path) -> list[dict]:
    """Читает CSV файл Турбоскрайб."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'start': int(row['start']),
                'end': int(row['end']),
                'text': row['text'].strip()
            })
    return rows


def merge_transcripts(part1_path: Path, part2_path: Path, offset_ms: int = OFFSET_MS) -> list[dict]:
    """Склеивает две части транскрипции с коррекцией таймкодов."""
    # Читаем первую часть
    part1 = read_csv(part1_path)

    # Читаем вторую часть и добавляем смещение
    part2 = read_csv(part2_path)
    for row in part2:
        row['start'] += offset_ms
        row['end'] += offset_ms

    # Объединяем
    merged = part1 + part2

    # Очищаем тексты
    for row in merged:
        row['text'] = clean_text(row['text'])

    # Фильтруем пустые строки
    merged = [r for r in merged if r['text']]

    return merged


def write_csv(rows: list[dict], filepath: Path):
    """Записывает CSV файл."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start', 'end', 'text'])
        writer.writeheader()
        writer.writerows(rows)


def write_txt(rows: list[dict], filepath: Path):
    """Записывает текст без таймкодов."""
    text = ' '.join(r['text'] for r in rows)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


def validate_merged(rows: list[dict], expected_duration_ms: int = None) -> dict:
    """Валидирует склеенный файл."""
    if not rows:
        return {'valid': False, 'error': 'Empty file'}

    # Проверка монотонности таймкодов
    prev_end = 0
    gaps = []
    overlaps = []
    for i, row in enumerate(rows):
        if row['start'] < prev_end - 100:  # допуск 100мс
            overlaps.append((i, prev_end - row['start']))
        elif row['start'] > prev_end + 5000:  # пауза > 5с
            gaps.append((i, row['start'] - prev_end))
        prev_end = row['end']

    result = {
        'valid': True,
        'total_rows': len(rows),
        'duration_ms': rows[-1]['end'],
        'duration_str': f"{rows[-1]['end'] // 60000}:{(rows[-1]['end'] % 60000) // 1000:02d}",
        'total_chars': sum(len(r['text']) for r in rows),
        'overlaps': len(overlaps),
        'large_gaps': len(gaps),
    }

    if expected_duration_ms:
        diff = abs(result['duration_ms'] - expected_duration_ms)
        result['duration_diff_ms'] = diff
        if diff > 60000:  # отличие > 1 минуты
            result['warning'] = f"Duration differs by {diff/1000:.1f}s from expected"

    return result


def process_chapter(chapter_num: int, input_dir: Path, output_dir: Path,
                    expected_duration_ms: int = None) -> dict:
    """Обрабатывает одну главу."""
    # Находим файлы
    part1_pattern = f"{chapter_num:02d}*yandex.csv"
    part2_pattern = f"{chapter_num:02d}*окончание*.csv"

    part1_files = list(input_dir.glob(part1_pattern))
    part2_files = list(input_dir.glob(part2_pattern))

    if not part1_files:
        return {'error': f'Part 1 not found: {part1_pattern}'}
    if not part2_files:
        return {'error': f'Part 2 not found: {part2_pattern}'}

    part1_path = part1_files[0]
    part2_path = part2_files[0]

    # Склеиваем
    merged = merge_transcripts(part1_path, part2_path)

    # Записываем
    output_csv = output_dir / f"{chapter_num:02d}_merged.csv"
    output_txt = output_dir / f"{chapter_num:02d}_merged.txt"

    write_csv(merged, output_csv)
    write_txt(merged, output_txt)

    # Валидируем
    validation = validate_merged(merged, expected_duration_ms)
    validation['input_part1'] = part1_path.name
    validation['input_part2'] = part2_path.name
    validation['output_csv'] = output_csv.name
    validation['output_txt'] = output_txt.name

    return validation


def main():
    """CLI интерфейс."""
    import argparse

    parser = argparse.ArgumentParser(description='Merge Turboscribe CSV files')
    parser.add_argument('--chapters', type=int, nargs='+', default=[1, 2, 3],
                        help='Chapter numbers to process')
    parser.add_argument('--input-dir', type=Path,
                        default=Path('Транскрибации/Турбоскрайб'),
                        help='Input directory')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('Транскрибации/Турбоскрайб'),
                        help='Output directory')

    args = parser.parse_args()

    # Ожидаемые длительности (из ffprobe)
    expected_durations = {
        1: 3611226,  # ~60 минут
        2: 2277335,  # ~38 минут
        3: 3127073,  # ~52 минуты
    }

    print(f"Turboscribe Merger v{VERSION}")
    print("=" * 50)

    results = {}
    for chapter in args.chapters:
        chapter_dir = args.input_dir / f"Глава {chapter}"
        if not chapter_dir.exists():
            print(f"Глава {chapter}: директория не найдена")
            continue

        print(f"\nГлава {chapter}:")
        result = process_chapter(
            chapter,
            chapter_dir,
            args.output_dir,
            expected_durations.get(chapter)
        )
        results[chapter] = result

        if 'error' in result:
            print(f"  ❌ Ошибка: {result['error']}")
        else:
            print(f"  ✓ Склеено: {result['total_rows']} сегментов")
            print(f"  ✓ Длительность: {result['duration_str']}")
            print(f"  ✓ Символов: {result['total_chars']}")
            if result['overlaps']:
                print(f"  ⚠ Перекрытий: {result['overlaps']}")
            if result['large_gaps']:
                print(f"  ⚠ Больших пауз: {result['large_gaps']}")
            if 'warning' in result:
                print(f"  ⚠ {result['warning']}")

    print("\n" + "=" * 50)
    print("Готово!")

    return results


if __name__ == '__main__':
    main()
