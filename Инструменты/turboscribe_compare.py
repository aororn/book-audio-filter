#!/usr/bin/env python3
"""
Turboscribe Compare v1.0
Сравнивает транскрипцию Турбоскрайб с оригиналом.

Использует тот же алгоритм выравнивания, что и smart_compare.py,
но принимает на вход нормализованные тексты напрямую.
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from filters import should_filter_error, normalize_word

VERSION = "1.0.0"


def load_text(filepath: Path) -> list[str]:
    """Загружает текст и разбивает на слова."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Нормализуем и разбиваем на слова
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    words = text.split()

    return words


def align_texts(transcript_words: list[str], original_words: list[str]) -> list[dict]:
    """
    Выравнивает два текста и находит различия.
    Возвращает список ошибок в формате compared.json.
    """
    errors = []

    # Используем SequenceMatcher для выравнивания
    matcher = SequenceMatcher(None, transcript_words, original_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        if tag == 'replace':
            # Замена слов
            for k, (t_idx, o_idx) in enumerate(zip(range(i1, i2), range(j1, j2))):
                transcript_word = transcript_words[t_idx]
                original_word = original_words[o_idx]

                # Пропускаем идентичные слова после нормализации
                if normalize_word(transcript_word) == normalize_word(original_word):
                    continue

                errors.append({
                    'type': 'substitution',
                    'transcript_word': transcript_word,
                    'original_word': original_word,
                    'position': o_idx,
                    'context_before': ' '.join(original_words[max(0, o_idx-3):o_idx]),
                    'context_after': ' '.join(original_words[o_idx+1:o_idx+4]),
                })

            # Если замена неравной длины — добавляем остаток
            if i2 - i1 > j2 - j1:
                # Лишние слова в транскрипции (insertion)
                for t_idx in range(i1 + (j2 - j1), i2):
                    errors.append({
                        'type': 'insertion',
                        'transcript_word': transcript_words[t_idx],
                        'original_word': '',
                        'position': j2,
                        'context_before': ' '.join(original_words[max(0, j2-3):j2]),
                        'context_after': ' '.join(original_words[j2:j2+3]),
                    })
            elif j2 - j1 > i2 - i1:
                # Пропущенные слова (deletion)
                for o_idx in range(j1 + (i2 - i1), j2):
                    errors.append({
                        'type': 'deletion',
                        'transcript_word': '',
                        'original_word': original_words[o_idx],
                        'position': o_idx,
                        'context_before': ' '.join(original_words[max(0, o_idx-3):o_idx]),
                        'context_after': ' '.join(original_words[o_idx+1:o_idx+4]),
                    })

        elif tag == 'delete':
            # Слова есть в транскрипции, но нет в оригинале (insertion)
            for t_idx in range(i1, i2):
                errors.append({
                    'type': 'insertion',
                    'transcript_word': transcript_words[t_idx],
                    'original_word': '',
                    'position': j1,
                    'context_before': ' '.join(original_words[max(0, j1-3):j1]),
                    'context_after': ' '.join(original_words[j1:j1+3]),
                })

        elif tag == 'insert':
            # Слова есть в оригинале, но нет в транскрипции (deletion)
            for o_idx in range(j1, j2):
                errors.append({
                    'type': 'deletion',
                    'transcript_word': '',
                    'original_word': original_words[o_idx],
                    'position': o_idx,
                    'context_before': ' '.join(original_words[max(0, o_idx-3):o_idx]),
                    'context_after': ' '.join(original_words[o_idx+1:o_idx+4]),
                })

    return errors


def filter_errors(errors: list[dict]) -> tuple[list[dict], list[dict]]:
    """Применяет фильтры к ошибкам."""
    filtered = []
    kept = []

    for error in errors:
        should_filter, reason = should_filter_error(error)
        error['filtered'] = should_filter
        error['filter_reason'] = reason

        if should_filter:
            filtered.append(error)
        else:
            kept.append(error)

    return kept, filtered


def calculate_wer(errors: list[dict], total_words: int) -> float:
    """Вычисляет Word Error Rate."""
    if total_words == 0:
        return 0.0
    return len(errors) / total_words


def compare_chapter(transcript_path: Path, original_path: Path, output_dir: Path) -> dict:
    """Сравнивает одну главу."""
    print(f"\nЗагрузка файлов...")

    # Загружаем тексты
    transcript_words = load_text(transcript_path)
    original_words = load_text(original_path)

    print(f"  Транскрипция: {len(transcript_words)} слов")
    print(f"  Оригинал: {len(original_words)} слов")

    # Выравниваем и находим ошибки
    print("Выравнивание текстов...")
    errors = align_texts(transcript_words, original_words)
    print(f"  Найдено различий: {len(errors)}")

    # Применяем фильтры
    print("Применение фильтров...")
    kept, filtered = filter_errors(errors)
    print(f"  После фильтрации: {len(kept)}")
    print(f"  Отфильтровано: {len(filtered)}")

    # Подсчёт по типам
    by_type = {}
    for e in kept:
        t = e['type']
        by_type[t] = by_type.get(t, 0) + 1

    # WER
    wer_raw = calculate_wer(errors, len(original_words))
    wer_filtered = calculate_wer(kept, len(original_words))

    # Формируем результат
    chapter_name = transcript_path.stem.replace('_normalized', '')

    result = {
        'metadata': {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'transcript_file': transcript_path.name,
            'original_file': original_path.name,
            'transcript_words': len(transcript_words),
            'original_words': len(original_words),
        },
        'summary': {
            'total_errors': len(errors),
            'filtered_errors': len(filtered),
            'remaining_errors': len(kept),
            'by_type': by_type,
            'wer_raw': round(wer_raw * 100, 2),
            'wer_filtered': round(wer_filtered * 100, 2),
        },
        'errors': kept,
        'filtered_errors': filtered,
    }

    # Сохраняем
    compared_path = output_dir / f"{chapter_name}_compared.json"
    filtered_path = output_dir / f"{chapter_name}_filtered.json"

    with open(compared_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Отдельный файл с отфильтрованными
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump({'filtered_errors': filtered}, f, ensure_ascii=False, indent=2)

    print(f"\n  Сохранено: {compared_path.name}")
    print(f"  Сохранено: {filtered_path.name}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare Turboscribe transcript with original')
    parser.add_argument('--chapters', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--transcript-dir', type=Path,
                        default=Path('Транскрибации/Турбоскрайб'))
    parser.add_argument('--original-dir', type=Path,
                        default=Path('Результаты проверки'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('Транскрибации/Турбоскрайб'))

    args = parser.parse_args()

    print(f"Turboscribe Compare v{VERSION}")
    print("=" * 50)

    results = {}

    for chapter in args.chapters:
        print(f"\n{'='*50}")
        print(f"ГЛАВА {chapter}")
        print("=" * 50)

        transcript_path = args.transcript_dir / f"{chapter:02d}_normalized.txt"

        # Ищем оригинал книги (не транскрипцию Яндекса!)
        original_candidates = [
            args.transcript_dir / f"original_{chapter:02d}_normalized.txt",
        ]

        original_path = None
        for candidate in original_candidates:
            if candidate.exists():
                original_path = candidate
                break

        if not transcript_path.exists():
            print(f"  ❌ Транскрипция не найдена: {transcript_path}")
            continue

        if original_path is None:
            print(f"  ❌ Оригинал не найден")
            continue

        print(f"  Транскрипция: {transcript_path.name}")
        print(f"  Оригинал: {original_path.name}")

        result = compare_chapter(transcript_path, original_path, args.output_dir)
        results[chapter] = result

    # Итоговая сводка
    print("\n" + "=" * 50)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 50)

    total_errors = 0
    total_filtered = 0
    total_words = 0

    for chapter, result in results.items():
        s = result['summary']
        print(f"\nГлава {chapter}:")
        print(f"  Ошибок до фильтра: {s['total_errors']}")
        print(f"  Ошибок после фильтра: {s['remaining_errors']}")
        print(f"  WER raw: {s['wer_raw']:.2f}%")
        print(f"  WER filtered: {s['wer_filtered']:.2f}%")
        print(f"  По типам: {s['by_type']}")

        total_errors += s['remaining_errors']
        total_filtered += s['filtered_errors']
        total_words += result['metadata']['original_words']

    if results:
        print(f"\nВСЕГО:")
        print(f"  Ошибок: {total_errors}")
        print(f"  Отфильтровано: {total_filtered}")
        print(f"  WER: {total_errors / total_words * 100:.2f}%")

    return results


if __name__ == '__main__':
    main()
