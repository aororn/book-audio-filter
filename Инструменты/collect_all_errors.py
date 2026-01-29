#!/usr/bin/env python3
"""
Collect All Errors v1.0 — Сбор ВСЕХ ошибок со ВСЕХ транскрипций

Прогоняет smart_compare + engine на всех транскрипциях,
собирает ошибки в БД с информацией о фильтрах.

Использование:
    python collect_all_errors.py --rebuild   # Полная пересборка БД
    python collect_all_errors.py --chapter 01  # Только одна глава
    python collect_all_errors.py --stats     # Показать статистику

Changelog:
    v1.0 (2026-01-29): Начальная версия
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-29'

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Пути проекта
PROJECT_ROOT = Path(__file__).parent.parent
TRANSCRIPTIONS_DIR = PROJECT_ROOT / 'Транскрибации'
ORIGINALS_DIR = PROJECT_ROOT / 'Оригинал' / 'Главы'
TESTS_DIR = PROJECT_ROOT / 'Тесты'
TEMP_DIR = PROJECT_ROOT / 'Темп'

# Импорт модулей проекта
try:
    from smart_compare import smart_compare
    from filters.engine import filter_errors, VERSION as ENGINE_VERSION
    from false_positives_db import FalsePositivesDB
    HAS_MODULES = True
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    HAS_MODULES = False


@dataclass
class TranscriptInfo:
    """Информация о транскрипции"""
    path: Path
    chapter: str  # "01", "02", "03", "04"
    name: str     # "yandex", "16kbps", etc.
    source_id: str  # "01_yandex", "01_16kbps"


def discover_transcripts() -> Dict[str, List[TranscriptInfo]]:
    """
    Обнаруживает все транскрипции по главам.

    Returns:
        {"01": [TranscriptInfo, ...], "02": [...], ...}
    """
    transcripts: Dict[str, List[TranscriptInfo]] = {}

    # Маппинг папок на номера глав
    chapter_mapping = {
        'Глава1': '01',
        'Глава2': '02',
        'Глава3': '03',
        'Глава4': '04',
    }

    for folder_name, chapter_id in chapter_mapping.items():
        folder = TRANSCRIPTIONS_DIR / folder_name
        if not folder.exists():
            continue

        transcripts[chapter_id] = []

        # Ищем JSON транскрипции
        for json_file in folder.glob('*.json'):
            # Пропускаем _compared и _filtered
            if '_compared' in json_file.name or '_filtered' in json_file.name:
                continue

            # Извлекаем имя транскрипции
            name = json_file.stem
            # Упрощаем имя: 01_16kbps_transcript_NEW_... -> 16kbps
            if 'transcript' in name.lower():
                parts = name.split('_')
                if len(parts) >= 2:
                    name = parts[1]  # 16kbps, 32kbps, yandex, etc.
                    if name.lower() == 'transcript':
                        name = parts[0]  # fallback to first part

            source_id = f"{chapter_id}_{name}"

            transcripts[chapter_id].append(TranscriptInfo(
                path=json_file,
                chapter=chapter_id,
                name=name,
                source_id=source_id,
            ))

    return transcripts


def load_golden_standard(chapter: str) -> Dict[str, Dict]:
    """
    Загружает golden standard для главы.

    Returns:
        {"wrong→correct": {"time": ..., "type": ...}, ...}
    """
    golden_files = [
        TESTS_DIR / f'золотой_стандарт_глава{int(chapter)}.json',
        TESTS_DIR / f'золотой_стандарт_глава_{chapter}.json',
    ]

    golden_keys: Dict[str, Dict] = {}

    for golden_file in golden_files:
        if golden_file.exists():
            with open(golden_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for err in data.get('errors', []):
                wrong = err.get('wrong', '')
                correct = err.get('correct', '')
                key = f"{wrong}→{correct}"
                golden_keys[key] = {
                    'time': err.get('time', ''),
                    'time_seconds': err.get('time_seconds'),
                    'type': err.get('type', 'substitution'),
                    'context': err.get('context', ''),
                }
            break

    return golden_keys


def get_original_path(chapter: str) -> Optional[Path]:
    """Возвращает путь к оригиналу главы."""
    patterns = [
        ORIGINALS_DIR / f'Глава {int(chapter)}.docx',
        ORIGINALS_DIR / f'Глава{int(chapter)}.docx',
        ORIGINALS_DIR / f'Глава{chapter}.docx',
    ]

    for path in patterns:
        if path.exists():
            return path
    return None


def process_transcript(
    transcript: TranscriptInfo,
    original_path: Path,
    golden_keys: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Обрабатывает одну транскрипцию через smart_compare + engine.

    Returns:
        (all_errors, filtered_errors, filter_stats)
    """
    # Шаг 1: Запускаем smart_compare
    compared_output = TEMP_DIR / f'_temp_{transcript.source_id}_compared.json'

    result = smart_compare(
        str(transcript.path),
        str(original_path),
        output_path=str(compared_output),
        threshold=0.7,
        phantom_seconds=-1,  # Отключаем phantom
        force=True,
    )

    if not compared_output.exists():
        print(f"  ✗ smart_compare не создал файл для {transcript.source_id}")
        return [], [], {}

    # Загружаем результат
    with open(compared_output, 'r', encoding='utf-8') as f:
        compared_data = json.load(f)

    all_errors = compared_data.get('errors', [])

    # Шаг 2: Запускаем фильтрацию
    filtered_errors, removed_errors, filter_stats = filter_errors(all_errors)

    # Создаём маппинг removed_errors по ключу
    removed_map: Dict[str, str] = {}
    for err in removed_errors:
        wrong = err.get('wrong', err.get('transcript', ''))
        correct = err.get('correct', err.get('original', ''))
        key = f"{wrong}→{correct}"
        reason = err.get('filter_reason', 'unknown')
        removed_map[key] = reason

    # Добавляем filter_reason к all_errors
    for err in all_errors:
        wrong = err.get('wrong', err.get('transcript', ''))
        correct = err.get('correct', err.get('original', ''))
        key = f"{wrong}→{correct}"

        if key in removed_map:
            err['actual_filter'] = removed_map[key]
        else:
            err['actual_filter'] = None  # Не был отфильтрован

        # Помечаем golden
        err['is_golden'] = key in golden_keys

    # Удаляем временный файл
    compared_output.unlink(missing_ok=True)

    return all_errors, filtered_errors, filter_stats


def collect_all_errors(
    chapters: Optional[List[str]] = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Собирает все ошибки со всех транскрипций.

    Args:
        chapters: Список глав для обработки (None = все)
        rebuild: Очистить БД перед сбором
        verbose: Выводить прогресс

    Returns:
        Статистика сбора
    """
    if not HAS_MODULES:
        return {'error': 'Модули не загружены'}

    # Инициализируем БД
    db = FalsePositivesDB()

    if rebuild:
        count = db.clear_all()
        if verbose:
            print(f"✓ БД очищена ({count} паттернов удалено)")

    # Обнаруживаем транскрипции
    all_transcripts = discover_transcripts()

    if chapters:
        all_transcripts = {ch: tr for ch, tr in all_transcripts.items() if ch in chapters}

    stats = {
        'chapters_processed': 0,
        'transcripts_processed': 0,
        'total_errors': 0,
        'total_golden': 0,
        'total_fp': 0,
        'by_chapter': {},
    }

    # Версии для метаданных
    source_versions = {
        'smart_compare': '10.5.0',
        'engine': ENGINE_VERSION,
        'collector': VERSION,
    }

    for chapter_id, transcripts in sorted(all_transcripts.items()):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Глава {chapter_id}: {len(transcripts)} транскрипций")
            print(f"{'='*60}")

        # Загружаем golden и оригинал
        golden_keys = load_golden_standard(chapter_id)
        original_path = get_original_path(chapter_id)

        if not original_path:
            print(f"  ⚠ Оригинал для главы {chapter_id} не найден, пропускаю")
            continue

        chapter_stats = {
            'transcripts': len(transcripts),
            'errors': 0,
            'golden_found': 0,
            'fp_found': 0,
        }

        for transcript in transcripts:
            if verbose:
                print(f"\n  Обработка: {transcript.source_id}")

            all_errors, filtered_errors, filter_stats = process_transcript(
                transcript, original_path, golden_keys
            )

            # Добавляем в БД
            for err in all_errors:
                wrong = err.get('wrong', err.get('transcript', ''))
                correct = err.get('correct', err.get('original', ''))
                error_type = err.get('type', 'substitution')
                context = err.get('context', '')
                time_seconds = err.get('time_seconds')
                actual_filter = err.get('actual_filter')

                db.add_error_with_filter(
                    wrong=wrong,
                    correct=correct,
                    error_type=error_type,
                    source=transcript.source_id,
                    actual_filter=actual_filter,
                    chapter=chapter_id,
                    context=context,
                    time_seconds=time_seconds,
                    source_versions=source_versions,
                )

                chapter_stats['errors'] += 1
                if err.get('is_golden'):
                    chapter_stats['golden_found'] += 1
                else:
                    chapter_stats['fp_found'] += 1

            stats['transcripts_processed'] += 1

            if verbose:
                print(f"    Ошибок: {len(all_errors)}, после фильтра: {len(filtered_errors)}")

        # Обновляем стабильность для главы
        db.update_stability(chapter_id, len(transcripts))

        stats['chapters_processed'] += 1
        stats['total_errors'] += chapter_stats['errors']
        stats['total_golden'] += chapter_stats['golden_found']
        stats['total_fp'] += chapter_stats['fp_found']
        stats['by_chapter'][chapter_id] = chapter_stats

    # Размечаем golden в БД
    db.mark_golden()

    # Обновляем морфологию
    if verbose:
        print("\n\nОбновление морфологических признаков...")
    db.update_morphology()

    db.close()

    return stats


def show_stats():
    """Показывает текущую статистику БД."""
    db = FalsePositivesDB()

    print("\n=== Статистика БД ===")

    # Общая статистика
    basic = db.get_stats()
    print(f"\nВсего паттернов: {basic['total_patterns']}")
    print(f"Вхождений: {basic['total_occurrences']}")

    # ML-статистика
    ml = db.get_ml_stats()
    print(f"\nGolden (реальные ошибки): {ml['golden_count']}")
    print(f"Non-golden (FP): {ml['non_golden_count']}")
    print(f"Баланс классов: {ml['class_balance']}")

    # Точность фильтров
    filter_acc = db.get_filter_accuracy()
    print(f"\n=== Точность фильтров ===")
    print(f"FP паттернов: {filter_acc['total_fp']}")
    print(f"Отфильтровано: {filter_acc['filtered_fp']}")
    print(f"Пропущено: {filter_acc['unfiltered_fp']}")
    print(f"Recall: {filter_acc['recall']:.1%}")

    if filter_acc['by_filter']:
        print(f"\nТоп фильтров:")
        for flt, cnt in list(filter_acc['by_filter'].items())[:10]:
            print(f"  {flt}: {cnt}")

    if filter_acc['unfiltered_by_category']:
        print(f"\nНеотфильтрованные по категориям:")
        for cat, cnt in filter_acc['unfiltered_by_category'].items():
            print(f"  {cat}: {cnt}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Сбор всех ошибок со всех транскрипций',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--version', '-V', action='store_true', help='Версия')
    parser.add_argument('--rebuild', '-r', action='store_true',
                       help='Полная пересборка БД (очищает все данные)')
    parser.add_argument('--chapter', '-c', type=str,
                       help='Обработать только одну главу (01, 02, 03, 04)')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Показать статистику БД')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Тихий режим')

    args = parser.parse_args()

    if args.version:
        print(f"Collect All Errors v{VERSION} ({VERSION_DATE})")
        return

    if args.stats:
        show_stats()
        return

    # Сбор данных
    chapters = [args.chapter] if args.chapter else None

    print(f"\n{'='*60}")
    print(f"  Collect All Errors v{VERSION}")
    print(f"  Режим: {'пересборка' if args.rebuild else 'добавление'}")
    print(f"  Главы: {chapters if chapters else 'все'}")
    print(f"{'='*60}")

    stats = collect_all_errors(
        chapters=chapters,
        rebuild=args.rebuild,
        verbose=not args.quiet,
    )

    if 'error' in stats:
        print(f"\n✗ Ошибка: {stats['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  ИТОГО")
    print(f"{'='*60}")
    print(f"  Глав обработано: {stats['chapters_processed']}")
    print(f"  Транскрипций: {stats['transcripts_processed']}")
    print(f"  Всего ошибок: {stats['total_errors']}")
    print(f"  Golden: {stats['total_golden']}")
    print(f"  FP: {stats['total_fp']}")

    print("\n✓ Сбор завершён. Используйте --stats для просмотра статистики.")


if __name__ == '__main__':
    main()
