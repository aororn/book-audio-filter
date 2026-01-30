#!/usr/bin/env python3
"""
version_table.py — Унифицированная система версий и метрик.

Использование:
    python version_table.py                    # Таблица всех версий
    python version_table.py v5.7 v10.0 v11.5  # Сравнение конкретных версий
    python version_table.py --add              # Добавить текущую версию
    python version_table.py --current          # Показать текущие метрики

Версия: 1.0.0
Дата: 2026-01-30
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

VERSION = '1.0.0'

# Пути
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TESTS_DIR = PROJECT_DIR / 'Тесты'
VERSIONS_FILE = TESTS_DIR / 'versions.json'
RESULTS_DIR = PROJECT_DIR / 'Результаты проверки'


def load_versions() -> Dict[str, Any]:
    """Загрузить versions.json."""
    if not VERSIONS_FILE.exists():
        return {"_schema_version": "1.0", "versions": []}
    with open(VERSIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_versions(data: Dict[str, Any]) -> None:
    """Сохранить versions.json."""
    data['_updated'] = datetime.now().strftime('%Y-%m-%d')
    with open(VERSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Сохранено: {VERSIONS_FILE}")


def get_version_by_id(versions: List[Dict], version_id: str) -> Optional[Dict]:
    """Найти версию по ID (v5.7, v10.0 и т.д.)."""
    # Нормализуем ID
    vid = version_id.lower().strip()
    if not vid.startswith('v'):
        vid = 'v' + vid

    for v in versions:
        if v['id'].lower() == vid:
            return v
        # Частичное совпадение (v5.7 найдёт v5.7.2)
        if v['id'].lower().startswith(vid):
            return v
    return None


def print_table(versions: List[Dict], selected_ids: Optional[List[str]] = None) -> None:
    """Вывести таблицу версий."""
    if selected_ids:
        def matches(vid: str, sid: str) -> bool:
            vid = vid.lower()
            sid = sid.lower().lstrip('v')
            return vid == f'v{sid}' or vid.startswith(f'v{sid}.')

        versions = [v for v in versions if any(
            matches(v['id'], sid) for sid in selected_ids
        )]

    if not versions:
        print("Нет данных для отображения.")
        return

    # Заголовок
    print()
    print("=" * 85)
    print(f"{'Версия':<10} {'Дата':<12} {'Filter':<8} {'Гл.1':>6} {'Гл.2':>6} {'Гл.3':>6} {'Гл.4':>6} {'Всего':>7} {'Golden':>8} {'FP':>6}")
    print("-" * 85)

    for v in versions:
        vid = v['id']
        date = v.get('date', '?')
        fver = v.get('filter_version', '?')
        ch = v.get('chapters', {})
        totals = v.get('totals', {})

        ch1 = ch.get('1', {})
        ch2 = ch.get('2', {})
        ch3 = ch.get('3', {})
        ch4 = ch.get('4', {})

        ch1_str = str(ch1.get('total', '')) if ch1 else '—'
        ch2_str = str(ch2.get('total', '')) if ch2 else '—'
        ch3_str = str(ch3.get('total', '')) if ch3 else '—'
        ch4_str = str(ch4.get('total', '')) if ch4 else '—'

        total_err = totals.get('errors', '?')
        golden = totals.get('golden', '?')
        fp = totals.get('fp', '?')

        golden_str = f"{golden}/93" if golden != '?' else '?'

        print(f"{vid:<10} {date:<12} {fver:<8} {ch1_str:>6} {ch2_str:>6} {ch3_str:>6} {ch4_str:>6} {total_err:>7} {golden_str:>8} {fp:>6}")

    print("=" * 85)
    print()


def get_current_metrics() -> Dict[str, Any]:
    """Получить текущие метрики из filtered.json файлов."""
    chapters = {}
    total_errors = 0
    total_golden = 0

    # Загружаем golden стандарт для подсчёта
    golden_counts = {1: 31, 2: 21, 3: 20, 4: 21}  # Известные значения

    for i in range(1, 5):
        filtered_path = RESULTS_DIR / f'0{i}' / f'0{i}_filtered.json'
        if filtered_path.exists():
            with open(filtered_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            errors = data.get('errors', [])
            err_count = len(errors)
            chapters[str(i)] = {
                'total': err_count,
                'golden': golden_counts.get(i, 0)
            }
            total_errors += err_count
            total_golden += golden_counts.get(i, 0)

    return {
        'chapters': chapters,
        'totals': {
            'errors': total_errors,
            'golden': total_golden,
            'fp': total_errors - total_golden
        }
    }


def get_filter_version() -> str:
    """Получить текущую версию фильтра."""
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from filters.engine import VERSION
        return VERSION
    except ImportError:
        return '?'


def add_current_version(version_id: str, notes: str = '') -> None:
    """Добавить текущую версию в versions.json."""
    data = load_versions()
    metrics = get_current_metrics()
    filter_ver = get_filter_version()

    new_version = {
        'id': version_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'filter_version': filter_ver,
        'chapters': metrics['chapters'],
        'totals': metrics['totals'],
        'notes': notes or f'Added via version_table.py'
    }

    # Проверяем, нет ли уже такой версии
    existing = get_version_by_id(data['versions'], version_id)
    if existing:
        print(f"[WARN] Версия {version_id} уже существует. Обновляю...")
        idx = data['versions'].index(existing)
        data['versions'][idx] = new_version
    else:
        data['versions'].append(new_version)

    save_versions(data)
    print(f"[OK] Добавлена версия {version_id}")
    print_table([new_version])


def show_current() -> None:
    """Показать текущие метрики без сохранения."""
    metrics = get_current_metrics()
    filter_ver = get_filter_version()

    print()
    print("=" * 50)
    print(f"  ТЕКУЩИЕ МЕТРИКИ (filter v{filter_ver})")
    print("=" * 50)

    for ch_num, ch_data in sorted(metrics['chapters'].items()):
        print(f"  Глава {ch_num}: {ch_data['total']} ошибок (golden: {ch_data['golden']})")

    print("-" * 50)
    t = metrics['totals']
    print(f"  ИТОГО: {t['errors']} ошибок | Golden: {t['golden']}/93 | FP: {t['fp']}")
    print("=" * 50)
    print()


def main():
    args = sys.argv[1:]

    if not args:
        # Показать все версии
        data = load_versions()
        print_table(data.get('versions', []))
        return

    if args[0] == '--current':
        show_current()
        return

    if args[0] == '--add':
        if len(args) < 2:
            print("Использование: python version_table.py --add v12.0 [notes]")
            return
        version_id = args[1]
        notes = ' '.join(args[2:]) if len(args) > 2 else ''
        add_current_version(version_id, notes)
        return

    if args[0] == '--help' or args[0] == '-h':
        print(__doc__)
        return

    # Иначе — показать выбранные версии
    data = load_versions()
    print_table(data.get('versions', []), args)


if __name__ == '__main__':
    main()
