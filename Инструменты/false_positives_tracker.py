#!/usr/bin/env python3
"""
false_positives_tracker.py v1.0

Инструмент для накопительного учёта ложных срабатываний фильтра.
Собирает статистику по ложным ошибкам из разных транскрибаций,
группирует по паттернам и помогает приоритизировать работу над фильтрами.

Использование:
    # Добавить ложные срабатывания из отчёта
    python false_positives_tracker.py add 01_filtered.json --source "01_A"

    # Добавить с автоисключением золотого стандарта
    python false_positives_tracker.py add 01_filtered.json --golden Тесты/золотой_стандарт_глава1.json

    # Показать статистику
    python false_positives_tracker.py stats

    # Показать топ-20 самых частых ложных
    python false_positives_tracker.py top 20

    # Пометить паттерн как решённый
    python false_positives_tracker.py resolve "своим→свои"

    # Экспорт для анализа
    python false_positives_tracker.py export --format csv

    # Проверить регрессии (решённые паттерны, которые снова появились)
    python false_positives_tracker.py check-regressions 01_filtered.json

Workflow:
    1. После каждой транскрибации добавляем ложные: tracker add report.json
    2. Анализируем топ частых: tracker top 20
    3. Добавляем паттерн в фильтр
    4. Помечаем как решённый: tracker resolve "pattern"
    5. Периодически проверяем регрессии
"""

import json
import sys
import os
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

# Добавляем путь к инструментам
sys.path.insert(0, str(Path(__file__).parent))

# Импортируем конфигурацию
try:
    from config import PROJECT_DIR, DICTIONARIES_DIR
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    PROJECT_DIR = Path(__file__).parent.parent
    DICTIONARIES_DIR = PROJECT_DIR / 'Словари'

# Путь к файлу трекера
TRACKER_FILE = DICTIONARIES_DIR / 'false_positives_tracker.json'

# Версия формата
FORMAT_VERSION = "1.0"


def load_tracker() -> Dict[str, Any]:
    """Загружает данные трекера из файла."""
    if not TRACKER_FILE.exists():
        return create_empty_tracker()

    with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Миграция старых версий если нужно
    if data.get('format_version') != FORMAT_VERSION:
        data = migrate_tracker(data)

    return data


def save_tracker(data: Dict[str, Any]):
    """Сохраняет данные трекера в файл."""
    data['metadata']['last_updated'] = datetime.now().isoformat()

    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_empty_tracker() -> Dict[str, Any]:
    """Создаёт пустую структуру трекера."""
    return {
        'format_version': FORMAT_VERSION,
        'metadata': {
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_entries': 0,
            'resolved_count': 0,
            'sources_processed': []
        },
        'false_positives': {},  # key: "wrong→correct", value: entry
        'resolved': {}  # key: "wrong→correct", value: resolution info
    }


def migrate_tracker(data: Dict[str, Any]) -> Dict[str, Any]:
    """Мигрирует данные из старых версий формата."""
    # Пока нет старых версий, просто возвращаем как есть
    data['format_version'] = FORMAT_VERSION
    return data


def make_key(wrong: str, correct: str) -> str:
    """Создаёт уникальный ключ для пары ошибки."""
    return f"{wrong.lower()}→{correct.lower()}"


def load_golden_standard(path: Path) -> set:
    """Загружает золотой стандарт и возвращает set ключей."""
    if not path.exists():
        return set()

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    golden_keys = set()
    errors = data if isinstance(data, list) else data.get('errors', [])

    for err in errors:
        # Поддержка разных форматов: wrong/correct, spoken/expected, transcript/original
        wrong = err.get('wrong', err.get('spoken', err.get('transcript', '')))
        correct = err.get('correct', err.get('expected', err.get('original', '')))
        if wrong or correct:
            golden_keys.add(make_key(wrong, correct))

    return golden_keys


def load_filtered_report(path: Path) -> List[Dict]:
    """Загружает отфильтрованный отчёт с ошибками и нормализует поля."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    errors = data if isinstance(data, list) else data.get('errors', [])

    # Нормализуем поля: smart_compare использует original/transcript,
    # а golden_filter использует wrong/correct
    normalized = []
    for err in errors:
        normalized_err = dict(err)
        # Если есть original/transcript, копируем в wrong/correct
        if 'original' in err and 'wrong' not in err:
            normalized_err['correct'] = err.get('original', '')
        if 'transcript' in err and 'correct' not in err:
            normalized_err['wrong'] = err.get('transcript', '')
        normalized.append(normalized_err)

    return normalized


def add_false_positives(
    report_path: Path,
    source: str,
    golden_path: Optional[Path] = None
) -> Tuple[int, int, int]:
    """
    Добавляет ложные срабатывания из отчёта в трекер.

    Returns:
        (added, updated, skipped) - количество добавленных, обновлённых, пропущенных
    """
    tracker = load_tracker()

    # Загружаем золотой стандарт если указан
    golden_keys = set()
    if golden_path:
        golden_keys = load_golden_standard(golden_path)

    # Загружаем отчёт
    errors = load_filtered_report(report_path)

    added = 0
    updated = 0
    skipped = 0

    for err in errors:
        wrong = err.get('wrong', '')
        correct = err.get('correct', '')
        err_type = err.get('type', 'substitution')

        key = make_key(wrong, correct)

        # Пропускаем если это золотая ошибка (реальная ошибка чтеца)
        if key in golden_keys:
            skipped += 1
            continue

        # Пропускаем если уже решено
        if key in tracker['resolved']:
            skipped += 1
            continue

        if key in tracker['false_positives']:
            # Обновляем существующую запись
            entry = tracker['false_positives'][key]
            entry['count'] += 1
            if source not in entry['sources']:
                entry['sources'].append(source)
            entry['last_seen'] = datetime.now().isoformat()
            updated += 1
        else:
            # Создаём новую запись
            tracker['false_positives'][key] = {
                'wrong': wrong,
                'correct': correct,
                'type': err_type,
                'count': 1,
                'sources': [source],
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'time_seconds': err.get('time_seconds'),
                'context': err.get('context', '')[:100],
                'pattern_suggestion': suggest_pattern(wrong, correct, err_type),
                'notes': ''
            }
            added += 1

    # Обновляем метаданные
    tracker['metadata']['total_entries'] = len(tracker['false_positives'])
    if source not in tracker['metadata']['sources_processed']:
        tracker['metadata']['sources_processed'].append(source)

    save_tracker(tracker)

    return added, updated, skipped


def suggest_pattern(wrong: str, correct: str, err_type: str) -> str:
    """Предлагает категорию паттерна для ошибки."""
    wrong_lower = wrong.lower()
    correct_lower = correct.lower()

    # Пустые значения
    if not wrong or not correct:
        return "alignment_artifact" if err_type in ('insertion', 'deletion') else "unknown"

    # Одинаковые слова
    if wrong_lower == correct_lower:
        return "case_sensitivity"

    # Окончания
    if len(wrong) > 3 and len(correct) > 3:
        if wrong_lower[:-2] == correct_lower[:-2]:
            return "grammar_ending"
        if wrong_lower[:-3] == correct_lower[:-3]:
            return "grammar_ending"

    # Приставки
    prefixes = ['не', 'по', 'на', 'за', 'от', 'до', 'вы', 'у', 'с', 'в', 'из', 'при']
    for prefix in prefixes:
        if wrong_lower.startswith(prefix) and wrong_lower[len(prefix):] == correct_lower:
            return "prefix_variant"
        if correct_lower.startswith(prefix) and correct_lower[len(prefix):] == wrong_lower:
            return "prefix_variant"

    # Короткие слова
    if len(wrong) <= 2 or len(correct) <= 2:
        return "short_word"

    # Фонетическое сходство
    if are_phonetically_similar(wrong_lower, correct_lower):
        return "phonetic"

    # По умолчанию
    return "unknown"


def are_phonetically_similar(w1: str, w2: str) -> bool:
    """Проверяет фонетическое сходство слов."""
    # Простые фонетические замены
    phonetic_pairs = [
        ('е', 'и'), ('о', 'а'), ('г', 'в'), ('ч', 'щ'),
        ('тся', 'ться'), ('ого', 'ово'), ('его', 'ево'),
    ]

    for p1, p2 in phonetic_pairs:
        if w1.replace(p1, p2) == w2 or w2.replace(p1, p2) == w1:
            return True
        if w1.replace(p2, p1) == w2 or w2.replace(p2, p1) == w1:
            return True

    return False


def resolve_pattern(pattern: str, resolution_type: str = 'filter_added', notes: str = ''):
    """Помечает паттерн как решённый."""
    tracker = load_tracker()

    # Ищем по ключу или по части
    matching_keys = []
    for key in tracker['false_positives']:
        if pattern.lower() in key.lower():
            matching_keys.append(key)

    if not matching_keys:
        print(f"Паттерн не найден: {pattern}")
        return 0

    resolved_count = 0
    for key in matching_keys:
        entry = tracker['false_positives'].pop(key)
        tracker['resolved'][key] = {
            **entry,
            'resolved_at': datetime.now().isoformat(),
            'resolution_type': resolution_type,
            'resolution_notes': notes
        }
        resolved_count += 1

    tracker['metadata']['total_entries'] = len(tracker['false_positives'])
    tracker['metadata']['resolved_count'] = len(tracker['resolved'])

    save_tracker(tracker)

    return resolved_count


def check_regressions(report_path: Path) -> List[Dict]:
    """Проверяет регрессии — решённые паттерны, которые снова появились."""
    tracker = load_tracker()
    errors = load_filtered_report(report_path)

    regressions = []
    for err in errors:
        key = make_key(err.get('wrong', ''), err.get('correct', ''))
        if key in tracker['resolved']:
            regressions.append({
                'error': err,
                'resolved_info': tracker['resolved'][key]
            })

    return regressions


def get_top_false_positives(n: int = 20) -> List[Tuple[str, Dict]]:
    """Возвращает топ-N самых частых ложных срабатываний."""
    tracker = load_tracker()

    sorted_entries = sorted(
        tracker['false_positives'].items(),
        key=lambda x: (-x[1]['count'], x[0])
    )

    return sorted_entries[:n]


def get_stats() -> Dict[str, Any]:
    """Возвращает статистику трекера."""
    tracker = load_tracker()

    # Группировка по типам паттернов
    pattern_counts = defaultdict(int)
    type_counts = defaultdict(int)

    for entry in tracker['false_positives'].values():
        pattern_counts[entry.get('pattern_suggestion', 'unknown')] += 1
        type_counts[entry.get('type', 'unknown')] += 1

    return {
        'total_false_positives': len(tracker['false_positives']),
        'resolved_count': len(tracker['resolved']),
        'sources_processed': len(tracker['metadata'].get('sources_processed', [])),
        'by_pattern': dict(pattern_counts),
        'by_type': dict(type_counts),
        'created': tracker['metadata'].get('created'),
        'last_updated': tracker['metadata'].get('last_updated')
    }


def export_to_csv(output_path: Path):
    """Экспортирует ложные срабатывания в CSV."""
    tracker = load_tracker()

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'wrong', 'correct', 'type', 'count', 'sources',
            'pattern_suggestion', 'first_seen', 'context', 'notes'
        ])

        for key, entry in tracker['false_positives'].items():
            writer.writerow([
                entry['wrong'],
                entry['correct'],
                entry['type'],
                entry['count'],
                ';'.join(entry['sources']),
                entry.get('pattern_suggestion', ''),
                entry.get('first_seen', ''),
                entry.get('context', ''),
                entry.get('notes', '')
            ])


def export_to_json(output_path: Path):
    """Экспортирует ложные срабатывания в JSON."""
    tracker = load_tracker()

    export_data = {
        'exported_at': datetime.now().isoformat(),
        'stats': get_stats(),
        'false_positives': list(tracker['false_positives'].values())
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)


def print_stats():
    """Выводит статистику в консоль."""
    stats = get_stats()

    print("\n" + "="*60)
    print("СТАТИСТИКА ЛОЖНЫХ СРАБАТЫВАНИЙ")
    print("="*60)
    print(f"  Всего ложных:     {stats['total_false_positives']}")
    print(f"  Решено:           {stats['resolved_count']}")
    print(f"  Источников:       {stats['sources_processed']}")
    print(f"  Создан:           {stats['created']}")
    print(f"  Обновлён:         {stats['last_updated']}")

    print("\nПо типам паттернов:")
    for pattern, count in sorted(stats['by_pattern'].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    print("\nПо типам ошибок:")
    for err_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {err_type}: {count}")


def print_top(n: int):
    """Выводит топ ложных срабатываний."""
    top = get_top_false_positives(n)

    print(f"\n{'='*60}")
    print(f"ТОП-{n} ЛОЖНЫХ СРАБАТЫВАНИЙ")
    print("="*60)

    for i, (key, entry) in enumerate(top, 1):
        sources = ', '.join(entry['sources'][:3])
        if len(entry['sources']) > 3:
            sources += f" +{len(entry['sources'])-3}"

        print(f"\n{i}. {entry['wrong']} → {entry['correct']}")
        print(f"   Тип: {entry['type']}, Частота: {entry['count']}")
        print(f"   Паттерн: {entry.get('pattern_suggestion', '?')}")
        print(f"   Источники: {sources}")
        if entry.get('context'):
            ctx = entry['context'][:60]
            print(f"   Контекст: ...{ctx}...")


def main():
    parser = argparse.ArgumentParser(
        description='Трекер ложных срабатываний фильтра',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s add report.json --source "01_A"
  %(prog)s add report.json --golden golden.json --source "01_B"
  %(prog)s stats
  %(prog)s top 20
  %(prog)s resolve "своим→свои"
  %(prog)s export --format csv -o export.csv
  %(prog)s check-regressions report.json
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Команда')

    # add
    add_parser = subparsers.add_parser('add', help='Добавить ложные из отчёта')
    add_parser.add_argument('report', type=Path, help='Путь к filtered.json')
    add_parser.add_argument('--source', '-s', required=True, help='Идентификатор источника')
    add_parser.add_argument('--golden', '-g', type=Path, help='Путь к золотому стандарту')

    # stats
    subparsers.add_parser('stats', help='Показать статистику')

    # top
    top_parser = subparsers.add_parser('top', help='Топ частых ложных')
    top_parser.add_argument('n', type=int, nargs='?', default=20, help='Количество')

    # resolve
    resolve_parser = subparsers.add_parser('resolve', help='Пометить как решённый')
    resolve_parser.add_argument('pattern', help='Паттерн или часть ключа')
    resolve_parser.add_argument('--type', '-t', default='filter_added',
                                help='Тип решения: filter_added, wont_fix, etc.')
    resolve_parser.add_argument('--notes', '-n', default='', help='Заметки')

    # export
    export_parser = subparsers.add_parser('export', help='Экспорт данных')
    export_parser.add_argument('--format', '-f', choices=['csv', 'json'], default='json')
    export_parser.add_argument('--output', '-o', type=Path, help='Выходной файл')

    # check-regressions
    reg_parser = subparsers.add_parser('check-regressions', help='Проверить регрессии')
    reg_parser.add_argument('report', type=Path, help='Путь к filtered.json')

    args = parser.parse_args()

    if args.command == 'add':
        if not args.report.exists():
            print(f"Файл не найден: {args.report}")
            sys.exit(1)

        added, updated, skipped = add_false_positives(
            args.report, args.source, args.golden
        )
        print(f"\nДобавлено: {added}")
        print(f"Обновлено: {updated}")
        print(f"Пропущено: {skipped} (золотые или решённые)")
        print(f"Файл: {TRACKER_FILE}")

    elif args.command == 'stats':
        print_stats()

    elif args.command == 'top':
        print_top(args.n)

    elif args.command == 'resolve':
        count = resolve_pattern(args.pattern, args.type, args.notes)
        print(f"Решено паттернов: {count}")

    elif args.command == 'export':
        output = args.output
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d')
            ext = args.format
            output = PROJECT_DIR / f'false_positives_export_{timestamp}.{ext}'

        if args.format == 'csv':
            export_to_csv(output)
        else:
            export_to_json(output)

        print(f"Экспортировано в: {output}")

    elif args.command == 'check-regressions':
        if not args.report.exists():
            print(f"Файл не найден: {args.report}")
            sys.exit(1)

        regressions = check_regressions(args.report)

        if not regressions:
            print("\nРегрессий не найдено!")
        else:
            print(f"\n{'='*60}")
            print(f"РЕГРЕССИИ ({len(regressions)})")
            print("="*60)
            for reg in regressions:
                err = reg['error']
                info = reg['resolved_info']
                print(f"\n  {err.get('wrong')} → {err.get('correct')}")
                print(f"    Было решено: {info.get('resolved_at')}")
                print(f"    Тип решения: {info.get('resolution_type')}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
