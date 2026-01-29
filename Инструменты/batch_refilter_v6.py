#!/usr/bin/env python3
"""
Batch Refilter v6.0 - Массовая перефильтрация через версию 6.0

Прогоняет все compared.json через новый фильтр v6.0 и собирает статистику.

Использование:
    python batch_refilter_v6.py                    # Все главы
    python batch_refilter_v6.py --chapter 1       # Только глава 1
    python batch_refilter_v6.py --dry-run         # Только показать что будет обработано
    python batch_refilter_v6.py --compare-old     # Сравнить с предыдущими результатами
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Добавляем путь к модулям
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from filters import filter_report, __version__ as FILTER_VERSION


def find_compared_files(results_dir: Path) -> List[Path]:
    """Находит все compared.json файлы"""
    files = []
    for p in results_dir.rglob("*_compared.json"):
        files.append(p)
    # Также основные compared.json
    for p in results_dir.rglob("*/??_compared.json"):
        if p not in files:
            files.append(p)
    return sorted(files)


def find_old_filtered(compared_path: Path) -> Optional[Path]:
    """Находит старый filtered файл для сравнения"""
    # Пробуем разные варианты имён
    base = compared_path.stem.replace('_compared', '')
    parent = compared_path.parent

    candidates = [
        parent / f"{base}_filtered.json",
        parent / f"{base}_filtered_v57.json",
    ]

    for c in candidates:
        if c.exists():
            return c
    return None


def load_json(path: Path) -> Dict:
    """Загружает JSON файл"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    """Сохраняет JSON файл"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compare_results(old_errors: List[Dict], new_errors: List[Dict]) -> Dict:
    """Сравнивает старые и новые результаты фильтрации"""
    def error_key(e):
        """Уникальный ключ ошибки"""
        return (
            e.get('time', 0),
            e.get('type', ''),
            e.get('wrong', e.get('original', '')),
            e.get('correct', e.get('transcript', ''))
        )

    old_set = {error_key(e) for e in old_errors}
    new_set = {error_key(e) for e in new_errors}

    only_old = old_set - new_set  # Были раньше, нет теперь (новые FP)
    only_new = new_set - old_set  # Не было раньше, есть теперь (регрессии)

    return {
        'old_count': len(old_errors),
        'new_count': len(new_errors),
        'delta': len(new_errors) - len(old_errors),
        'new_false_positives': len(only_old),  # v6.0 отфильтровал больше
        'regressions': len(only_new),           # v6.0 пропустил что-то
    }


def extract_filter_stats(report: Dict) -> Dict:
    """Извлекает статистику фильтрации"""
    stats = report.get('filter_stats', {})
    meta = report.get('filter_metadata', {})

    return {
        'original_errors': meta.get('original_errors', 0),
        'filtered_errors': meta.get('filtered_errors', 0),
        'real_errors': meta.get('real_errors', 0),
        'efficiency': meta.get('filter_efficiency', '0%'),
        'by_rule': dict(stats),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Массовая перефильтрация v6.0')
    parser.add_argument('--chapter', '-c', type=int, help='Только указанная глава')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Только показать файлы')
    parser.add_argument('--compare-old', '-o', action='store_true', help='Сравнить со старыми результатами')
    parser.add_argument('--output-dir', '-O', type=Path, help='Папка для результатов')
    args = parser.parse_args()

    # Пути
    project_dir = SCRIPT_DIR.parent
    results_dir = project_dir / "Результаты проверки"
    output_dir = args.output_dir or results_dir / "v6_refilter"

    print(f"\n{'='*70}")
    print(f"  Batch Refilter v6.0 (filter version: {FILTER_VERSION})")
    print(f"{'='*70}")

    # Находим файлы
    compared_files = find_compared_files(results_dir)

    if args.chapter:
        compared_files = [f for f in compared_files if f'/{args.chapter:02d}/' in str(f) or f'0{args.chapter}_' in f.name]

    print(f"\nНайдено compared файлов: {len(compared_files)}")

    if args.dry_run:
        print("\n[DRY RUN] Файлы для обработки:")
        for f in compared_files:
            old = find_old_filtered(f)
            old_str = f" (old: {old.name})" if old else ""
            print(f"  - {f.relative_to(project_dir)}{old_str}")
        return

    # Создаём папку для результатов
    output_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем файлы
    all_stats = []
    total_rules = defaultdict(int)
    total_smart_rules = defaultdict(int)

    for compared_path in compared_files:
        print(f"\n--- Обработка: {compared_path.name} ---")

        try:
            report = load_json(compared_path)
        except Exception as e:
            print(f"  ✗ Ошибка загрузки: {e}")
            continue

        # Определяем имя выходного файла
        base = compared_path.stem.replace('_compared', '')
        chapter_dir = compared_path.parent.name
        out_name = f"{base}_filtered_v6.json"
        out_path = output_dir / chapter_dir
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / out_name

        # Фильтруем (filter_report принимает путь к файлу, не словарь)
        try:
            filtered_report = filter_report(
                report_path=str(compared_path),
                output_path=str(out_file),
                force=True
            )
        except Exception as e:
            print(f"  ✗ Ошибка фильтрации: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Статистика
        stats = extract_filter_stats(filtered_report)
        stats['file'] = compared_path.name
        stats['chapter'] = chapter_dir

        # Собираем статистику по правилам
        for rule, count in stats['by_rule'].items():
            total_rules[rule] += count
            if rule.startswith('smart_'):
                total_smart_rules[rule] += count

        # Сравнение со старым
        if args.compare_old:
            old_filtered_path = find_old_filtered(compared_path)
            if old_filtered_path:
                try:
                    old_report = load_json(old_filtered_path)
                    old_errors = old_report.get('errors', [])
                    new_errors = filtered_report.get('errors', [])
                    comparison = compare_results(old_errors, new_errors)
                    stats['comparison'] = comparison

                    delta_str = f"+{comparison['delta']}" if comparison['delta'] > 0 else str(comparison['delta'])
                    print(f"  Сравнение: {comparison['old_count']} → {comparison['new_count']} ({delta_str})")
                    if comparison['new_false_positives'] > 0:
                        print(f"    Новые FP (v6 отфильтровал): {comparison['new_false_positives']}")
                    if comparison['regressions'] > 0:
                        print(f"    ⚠ Регрессии (v6 пропустил): {comparison['regressions']}")
                except Exception as e:
                    print(f"  ✗ Ошибка сравнения: {e}")

        all_stats.append(stats)
        print(f"  ✓ {stats['original_errors']} → {stats['real_errors']} (эффективность: {stats['efficiency']})")

    # Итоговый отчёт
    print(f"\n{'='*70}")
    print(f"  ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*70}")

    total_original = sum(s['original_errors'] for s in all_stats)
    total_filtered = sum(s['filtered_errors'] for s in all_stats)
    total_real = sum(s['real_errors'] for s in all_stats)

    print(f"\n  Обработано файлов: {len(all_stats)}")
    print(f"  Всего ошибок на входе: {total_original}")
    print(f"  Отфильтровано: {total_filtered}")
    print(f"  Реальных ошибок: {total_real}")
    print(f"  Общая эффективность: {total_filtered / total_original * 100:.1f}%" if total_original > 0 else "")

    # Статистика по правилам
    print(f"\n  Топ-15 правил фильтрации:")
    for rule, count in sorted(total_rules.items(), key=lambda x: -x[1])[:15]:
        smart_marker = " ★" if rule.startswith('smart_') else ""
        print(f"    {rule}: {count}{smart_marker}")

    # Статистика smart_rules
    if total_smart_rules:
        print(f"\n  Smart Rules (v6.0):")
        for rule, count in sorted(total_smart_rules.items(), key=lambda x: -x[1]):
            print(f"    {rule}: {count}")
        total_smart = sum(total_smart_rules.values())
        print(f"  Всего smart_rules: {total_smart} ({total_smart / total_filtered * 100:.1f}% от отфильтрованных)")

    # Сравнение со старой версией
    if args.compare_old:
        comparisons = [s['comparison'] for s in all_stats if 'comparison' in s]
        if comparisons:
            total_old = sum(c['old_count'] for c in comparisons)
            total_new = sum(c['new_count'] for c in comparisons)
            total_new_fp = sum(c['new_false_positives'] for c in comparisons)
            total_regr = sum(c['regressions'] for c in comparisons)

            print(f"\n  Сравнение v5.7 → v6.0:")
            print(f"    Было ошибок: {total_old}")
            print(f"    Стало ошибок: {total_new}")
            print(f"    Новые FP (v6 отфильтровал больше): {total_new_fp}")
            print(f"    Регрессии (v6 пропустил): {total_regr}")

    # Сохраняем сводку
    summary = {
        'timestamp': datetime.now().isoformat(),
        'filter_version': FILTER_VERSION,
        'files_processed': len(all_stats),
        'totals': {
            'original_errors': total_original,
            'filtered_errors': total_filtered,
            'real_errors': total_real,
            'efficiency': f"{total_filtered / total_original * 100:.1f}%" if total_original > 0 else "0%",
        },
        'rules_stats': dict(total_rules),
        'smart_rules_stats': dict(total_smart_rules),
        'per_file': all_stats,
    }

    summary_path = output_dir / f"refilter_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    save_json(summary, summary_path)
    print(f"\n  Сводка сохранена: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
