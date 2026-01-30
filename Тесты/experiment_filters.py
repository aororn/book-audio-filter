#!/usr/bin/env python3
"""
Эксперименты с фильтрами v1.0

Тестирует различные конфигурации фильтров:
1. Снижение порога ML с 90% до 85%
2. Изменение порядка фильтров
3. A/B сравнение результатов

Использование:
    python Тесты/experiment_filters.py --ml-threshold 0.85
    python Тесты/experiment_filters.py --reorder-ml-first
    python Тесты/experiment_filters.py --compare

Версия: 1.0.0 (2026-01-30)
"""

VERSION = '1.0.0'

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Добавляем путь к модулям
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'Инструменты'))

from config import TESTS_DIR, TRANSCRIPTIONS_DIR, CHAPTERS_DIR

ANALYSIS_DIR = TESTS_DIR / 'Анализ_фильтров'
EXPERIMENTS_DIR = TESTS_DIR / 'Эксперименты'
GOLDEN_FILES = {
    '1': TESTS_DIR / 'золотой_стандарт_глава1.json',
    '2': TESTS_DIR / 'золотой_стандарт_глава2.json',
    '3': TESTS_DIR / 'золотой_стандарт_глава3.json',
    '4': TESTS_DIR / 'золотой_стандарт_глава4.json',
}


def load_golden_errors(chapter_num: str) -> List[Dict]:
    """Загружает golden ошибки для главы."""
    golden_file = GOLDEN_FILES.get(chapter_num)
    if not golden_file or not golden_file.exists():
        return []
    with open(golden_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('errors', data) if isinstance(data, dict) else data


def normalize_for_comparison(word: str) -> str:
    if not word:
        return ''
    return word.lower().strip().replace('ё', 'е')


def is_golden_error(error: Dict, golden_errors: List[Dict]) -> bool:
    """Проверяет, является ли ошибка golden."""
    error_type = error.get('type', '')

    if error_type == 'substitution':
        orig = normalize_for_comparison(error.get('original', '') or error.get('correct', ''))
        trans = normalize_for_comparison(error.get('transcript', '') or error.get('wrong', ''))
    elif error_type == 'insertion':
        orig = ''
        trans = normalize_for_comparison(error.get('transcript', '') or error.get('wrong', '') or error.get('word', ''))
    elif error_type == 'deletion':
        orig = normalize_for_comparison(error.get('original', '') or error.get('correct', '') or error.get('word', ''))
        trans = ''
    else:
        return False

    error_time = error.get('time', 0)
    try:
        error_time = float(error_time) if error_time else 0.0
    except (ValueError, TypeError):
        error_time = 0.0

    for golden in golden_errors:
        g_type = golden.get('type', '')
        if g_type != error_type:
            continue

        if g_type == 'substitution':
            g_orig = normalize_for_comparison(golden.get('original', '') or golden.get('correct', ''))
            g_trans = normalize_for_comparison(golden.get('transcript', '') or golden.get('wrong', ''))
        elif g_type == 'insertion':
            g_orig = ''
            g_trans = normalize_for_comparison(golden.get('transcript', '') or golden.get('wrong', '') or golden.get('word', ''))
        elif g_type == 'deletion':
            g_orig = normalize_for_comparison(golden.get('original', '') or golden.get('correct', '') or golden.get('word', ''))
            g_trans = ''
        else:
            continue

        if orig == g_orig and trans == g_trans:
            g_time = golden.get('time', 0)
            try:
                g_time = float(g_time) if g_time else 0.0
            except (ValueError, TypeError):
                g_time = 0.0
            if abs(error_time - g_time) <= 5:
                return True

    return False


def test_ml_threshold(threshold: float) -> Dict[str, Any]:
    """
    Тестирует ML-классификатор с заданным порогом.

    Возвращает статистику: сколько дополнительно отфильтровано,
    затронуты ли golden.
    """
    print(f"\n{'='*60}")
    print(f"  ЭКСПЕРИМЕНТ: ML порог = {threshold:.0%}")
    print(f"{'='*60}")

    # Импортируем ML-классификатор
    try:
        from ml_classifier import get_classifier
        classifier = get_classifier()
        if classifier.model is None:
            print("  ✗ ML модель не загружена")
            return {'error': 'model_not_loaded'}
    except Exception as e:
        print(f"  ✗ Ошибка загрузки ML: {e}")
        return {'error': str(e)}

    # Загружаем compared.json файлы из Анализ_фильтров
    results = {
        'threshold': threshold,
        'chapters': {},
        'total_additional_filtered': 0,
        'total_golden_hit': 0,
        'details': [],
    }

    for chapter_num in ['1', '2', '3', '4']:
        compared_file = ANALYSIS_DIR / f'{chapter_num.zfill(2)}_analysis_compared.json'
        if not compared_file.exists():
            print(f"  Глава {chapter_num}: файл не найден, пропуск")
            continue

        with open(compared_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        errors = data.get('errors', [])
        golden_errors = load_golden_errors(chapter_num)

        # Считаем, сколько дополнительно отфильтрует ML с новым порогом
        additional_filtered = 0
        golden_hit = 0

        for error in errors:
            if error.get('type') != 'substitution':
                continue

            orig = error.get('original', '')
            trans = error.get('transcript', '')
            if not orig or not trans:
                continue

            # Проверяем ML
            try:
                is_fp, confidence = classifier.predict(
                    normalize_for_comparison(trans),
                    normalize_for_comparison(orig)
                )
            except Exception:
                continue

            # Текущий порог 90% — что дополнительно отфильтруется при 85%?
            if is_fp and confidence >= threshold and confidence < 0.90:
                additional_filtered += 1
                is_golden = is_golden_error(error, golden_errors)
                if is_golden:
                    golden_hit += 1
                    results['details'].append({
                        'chapter': chapter_num,
                        'original': orig,
                        'transcript': trans,
                        'confidence': confidence,
                        'is_golden': True,
                    })
                else:
                    results['details'].append({
                        'chapter': chapter_num,
                        'original': orig,
                        'transcript': trans,
                        'confidence': confidence,
                        'is_golden': False,
                    })

        results['chapters'][chapter_num] = {
            'additional_filtered': additional_filtered,
            'golden_hit': golden_hit,
        }
        results['total_additional_filtered'] += additional_filtered
        results['total_golden_hit'] += golden_hit

        status = '✓' if golden_hit == 0 else f'⚠ {golden_hit} golden!'
        print(f"  Глава {chapter_num}: +{additional_filtered} FP, {status}")

    print(f"\n  ИТОГО:")
    print(f"    Дополнительно отфильтровано: +{results['total_additional_filtered']} FP")
    print(f"    Golden затронуто: {results['total_golden_hit']} {'✓' if results['total_golden_hit'] == 0 else '⚠ ОПАСНО!'}")

    if results['details']:
        print(f"\n  Детали (порог {threshold:.0%} - 90%):")
        for d in results['details'][:10]:
            status = '⚠ GOLDEN' if d['is_golden'] else 'FP'
            print(f"    [{status}] {d['original']} → {d['transcript']} ({d['confidence']:.0%})")
        if len(results['details']) > 10:
            print(f"    ... и ещё {len(results['details']) - 10}")

    print(f"{'='*60}\n")

    return results


def test_multiple_thresholds() -> Dict[str, Any]:
    """Тестирует несколько порогов ML."""
    thresholds = [0.85, 0.80, 0.75, 0.70]
    all_results = {}

    print(f"\n{'#'*60}")
    print(f"  ЭКСПЕРИМЕНТ: СРАВНЕНИЕ ПОРОГОВ ML")
    print(f"{'#'*60}")

    for threshold in thresholds:
        all_results[str(threshold)] = test_ml_threshold(threshold)

    # Сводная таблица
    print(f"\n{'='*60}")
    print(f"  СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*60}")
    print(f"\n  {'Порог':<10} {'Доп. FP':<12} {'Golden':<10} {'Статус'}")
    print(f"  {'-'*45}")

    for threshold in thresholds:
        res = all_results[str(threshold)]
        if 'error' in res:
            print(f"  {threshold*100:.0f}%       {'ошибка':<12}")
            continue
        status = '✓ Безопасно' if res['total_golden_hit'] == 0 else '⚠ ОПАСНО'
        print(f"  {threshold*100:.0f}%       +{res['total_additional_filtered']:<11} {res['total_golden_hit']:<10} {status}")

    print(f"{'='*60}\n")

    return all_results


def find_original_for_chapter(chapter_num: str) -> Optional[Path]:
    """Находит файл оригинала для главы."""
    variants = [
        f'Глава{chapter_num}.docx',
        f'Глава {chapter_num}.docx',
        f'Глава_{chapter_num}.docx',
    ]
    for variant in variants:
        path = CHAPTERS_DIR / variant
        if path.exists():
            return path
    return None


def test_filter_order_experiment() -> Dict[str, Any]:
    """
    Эксперимент: что если ML-классификатор запустить раньше morpho_rules?

    Гипотеза: ML может отфильтровать больше FP, если запустить его
    до морфологических правил.

    Метод: симулируем изменённый порядок, подсчитываем разницу.
    """
    print(f"\n{'='*60}")
    print(f"  ЭКСПЕРИМЕНТ: ПОРЯДОК ФИЛЬТРОВ")
    print(f"{'='*60}")
    print(f"  Гипотеза: ML раньше morpho_rules увеличит фильтрацию")

    try:
        from ml_classifier import get_classifier
        from filters.morpho_rules import get_morpho_rules
        from filters.comparison import normalize_word, get_lemma
        classifier = get_classifier()
        morpho = get_morpho_rules()
    except Exception as e:
        print(f"  ✗ Ошибка загрузки: {e}")
        return {'error': str(e)}

    results = {
        'experiment': 'filter_order',
        'chapters': {},
        'current_order': {'morpho_first': 0, 'ml_first': 0},
        'proposed_order': {'morpho_first': 0, 'ml_first': 0},
        'difference': 0,
    }

    # Считаем сколько ошибок фильтруется каждым способом
    for chapter_num in ['1', '2', '3', '4']:
        compared_file = ANALYSIS_DIR / f'{chapter_num.zfill(2)}_analysis_compared.json'
        if not compared_file.exists():
            continue

        with open(compared_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        errors = data.get('errors', [])
        golden_errors = load_golden_errors(chapter_num)

        morpho_only = 0
        ml_only = 0
        both = 0
        neither = 0

        for error in errors:
            if error.get('type') != 'substitution':
                continue

            orig = normalize_for_comparison(error.get('original', ''))
            trans = normalize_for_comparison(error.get('transcript', ''))
            if not orig or not trans:
                continue

            # Проверяем morpho
            morpho_result = morpho.check(trans, orig)
            morpho_filters = morpho_result and morpho_result.should_filter

            # Проверяем ML (порог 85%)
            try:
                is_fp, confidence = classifier.predict(trans, orig)
                ml_filters = is_fp and confidence >= 0.85
            except Exception:
                ml_filters = False

            if morpho_filters and ml_filters:
                both += 1
            elif morpho_filters:
                morpho_only += 1
            elif ml_filters:
                ml_only += 1
            else:
                neither += 1

        results['chapters'][chapter_num] = {
            'morpho_only': morpho_only,
            'ml_only': ml_only,
            'both': both,
            'neither': neither,
        }

        print(f"  Глава {chapter_num}:")
        print(f"    Только morpho: {morpho_only}")
        print(f"    Только ML: {ml_only}")
        print(f"    Оба: {both}")
        print(f"    Ни один: {neither}")

    # Итого
    total_morpho_only = sum(c['morpho_only'] for c in results['chapters'].values())
    total_ml_only = sum(c['ml_only'] for c in results['chapters'].values())
    total_both = sum(c['both'] for c in results['chapters'].values())

    print(f"\n  ИТОГО:")
    print(f"    Только morpho: {total_morpho_only}")
    print(f"    Только ML (85 pct): {total_ml_only}")
    print(f"    Перекрытие: {total_both}")

    # Анализ
    print(f"\n  АНАЛИЗ:")
    if total_ml_only > 0:
        print(f"    ML фильтрует {total_ml_only} ошибок, которые morpho пропускает")
        print(f"    → Изменение порядка НЕ НУЖНО (ML уже работает после morpho)")
    else:
        print(f"    ML не добавляет фильтраций сверх morpho")

    print(f"    Перекрытие {total_both} ошибок — порядок не важен для них")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=f'Эксперименты с фильтрами v{VERSION}',
    )
    parser.add_argument('--ml-threshold', type=float, default=0.85,
                        help='Тестовый порог ML (по умолчанию 0.85)')
    parser.add_argument('--all-thresholds', action='store_true',
                        help='Тестировать все пороги (85, 80, 75, 70 процентов)')
    parser.add_argument('--filter-order', action='store_true',
                        help='Эксперимент с порядком фильтров')
    parser.add_argument('--version', '-V', action='store_true',
                        help='Показать версию')

    args = parser.parse_args()

    if args.version:
        print(f"experiment_filters v{VERSION}")
        return 0

    # Создаём папку для результатов
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.filter_order:
        results = test_filter_order_experiment()
    elif args.all_thresholds:
        results = test_multiple_thresholds()
    else:
        results = test_ml_threshold(args.ml_threshold)

    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = EXPERIMENTS_DIR / f'ml_threshold_experiment_{timestamp}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Результаты сохранены: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
