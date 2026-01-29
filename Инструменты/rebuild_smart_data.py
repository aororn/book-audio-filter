#!/usr/bin/env python3
"""
Rebuild Smart Data v2.0 — Пересчёт SmartFilter данных для всех паттернов в БД.

v2.0: Использует новую систему весов:
    - Базовый score = 0
    - Морфология (леммы, POS) — ключевой признак
    - Длина слова — короткие = FP
    - Частотность — редкие = важные, частые = FP

Использование:
    python rebuild_smart_data.py              # пересчитать все паттерны
    python rebuild_smart_data.py --stats      # показать статистику
    python rebuild_smart_data.py --calibrate  # данные для калибровки
    python rebuild_smart_data.py --no-semantics  # без загрузки Word2Vec (быстрее)

Версия: 2.0.0
Дата: 2026-01-30
"""

import argparse
import json
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from false_positives_db import FalsePositivesDB
from filters.smart_filter import SmartFilter
from filters.frequency_manager import get_frequency_manager
from filters.sliding_window import get_sliding_window

VERSION = '2.0.0'


def rebuild_all(use_semantics: bool = True, verbose: bool = True):
    """
    Пересчитать SmartFilter данные для всех паттернов.

    Args:
        use_semantics: Загружать Word2Vec модель (231 МБ, ~5 сек)
        verbose: Выводить прогресс
    """
    print("=" * 60)
    print(f"  Rebuild Smart Data v{VERSION}")
    print("=" * 60)

    # Инициализация
    db = FalsePositivesDB()
    sf = SmartFilter(use_semantics=use_semantics, threshold=60)
    fm = get_frequency_manager()
    sw = get_sliding_window()

    # Семантика (опционально)
    sm = None
    if use_semantics:
        try:
            from filters.semantic_manager import get_semantic_manager
            sm = get_semantic_manager()
            print("  SemanticManager: загружен")
        except Exception as e:
            print(f"  SemanticManager: не загружен ({e})")
    else:
        print("  SemanticManager: отключен (--no-semantics)")

    # Получаем все паттерны
    patterns = db.get_all_patterns()
    total = len(patterns)
    print(f"  Паттернов в БД: {total}")
    print("=" * 60)

    # Счётчики
    updated = 0
    errors = 0

    for i, pattern in enumerate(patterns):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Обработано: {i + 1}/{total}")

        try:
            # Извлекаем данные
            wrong = pattern['wrong'] or ''
            correct = pattern['correct'] or ''
            error_type = pattern['error_type'] or 'substitution'
            is_golden = pattern['is_golden'] or 0

            # 1. Частотность
            freq1 = fm.get_frequency(wrong) if wrong else 0.0
            freq2 = fm.get_frequency(correct) if correct else 0.0

            # Категория частотности
            freq_category = 'unknown'
            check_word = correct if error_type == 'deletion' else wrong
            if check_word:
                freq_category = fm.get_category(check_word)

            # 2. Семантика (только substitution)
            semantic_sim = 0.0
            is_slip = 0
            if sm and error_type == 'substitution' and wrong and correct:
                try:
                    semantic_sim = sm.similarity(wrong, correct)
                    is_slip = 1 if semantic_sim > 0.6 else 0
                except Exception:
                    pass

            # 3. SlidingWindow (упрощённая проверка)
            sliding_match = 0
            sliding_sim = 0
            if wrong and correct:
                result = sw.check_artifact([wrong], [correct])
                sliding_match = 1 if result.is_artifact else 0
                sliding_sim = result.similarity

            # 4. SmartFilter скор
            error = {
                'type': error_type,
                'wrong': wrong,
                'correct': correct,
                'context': '',
            }
            sf_result = sf.evaluate_error(error)
            smart_score = sf_result.score
            smart_rules = json.dumps(sf_result.applied_rules, ensure_ascii=False)

            # 5. Валидация: score_correct
            # Golden должен иметь score >= 60
            # FP должен иметь score < 60
            if is_golden == 1:
                score_correct = 1 if smart_score >= 60 else 0
            else:
                score_correct = 1 if smart_score < 60 else 0

            # 6. Обновляем БД
            db.update_smart_data(pattern['id'], {
                'smart_score': smart_score,
                'smart_rules': smart_rules,
                'freq1': freq1,
                'freq2': freq2,
                'freq_category': freq_category,
                'semantic_similarity': semantic_sim,
                'is_semantic_slip': is_slip,
                'sliding_match': sliding_match,
                'sliding_similarity': sliding_sim,
                'score_correct': score_correct,
            })
            updated += 1

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  Ошибка паттерна {pattern['id']}: {e}")

    print("=" * 60)
    print(f"  Обновлено: {updated}")
    print(f"  Ошибок: {errors}")
    print("=" * 60)

    # Показываем статистику
    show_stats(db)


def show_stats(db: FalsePositivesDB = None):
    """Показать статистику SmartFilter."""
    if db is None:
        db = FalsePositivesDB()

    print("\n" + "=" * 60)
    print("  SmartFilter статистика")
    print("=" * 60)

    # Распределение скоров
    dist = db.get_score_distribution()
    print(f"\n  Golden (is_golden=1):")
    print(f"    Всего: {dist['golden']['total']}")
    print(f"    Score: min={dist['golden']['min']}, max={dist['golden']['max']}, avg={dist['golden']['avg']}")
    print(f"    Ниже порога 60: {dist['golden_below_threshold']} (ОШИБКИ!)")

    print(f"\n  FP (is_golden=0):")
    print(f"    Всего: {dist['fp']['total']}")
    print(f"    Score: min={dist['fp']['min']}, max={dist['fp']['max']}, avg={dist['fp']['avg']}")
    print(f"    Выше порога 60: {dist['fp_above_threshold']} (не отфильтрованы)")

    # Accuracy
    smart_stats = db.get_smart_stats()
    print(f"\n  Общая точность:")
    print(f"    С SmartFilter данными: {smart_stats['with_smart_score']}")
    print(f"    Корректных: {smart_stats['score_correct']}")
    print(f"    Некорректных: {smart_stats['score_incorrect']}")
    if smart_stats['with_smart_score'] > 0:
        accuracy = smart_stats['score_correct'] / smart_stats['with_smart_score'] * 100
        print(f"    Accuracy: {accuracy:.1f}%")

    # По категориям частотности
    print(f"\n  По категориям частотности:")
    for cat, count in smart_stats['by_frequency_category'].items():
        print(f"    {cat}: {count}")

    print("=" * 60)


def show_calibration():
    """Показать данные для калибровки весов."""
    db = FalsePositivesDB()
    data = db.get_calibration_data()

    print("\n" + "=" * 60)
    print("  Данные для калибровки весов")
    print("=" * 60)

    # Golden которые были бы отфильтрованы
    golden_filtered = data['golden_filtered']
    print(f"\n  Golden со score < 60 (ОШИБКИ!): {len(golden_filtered)}")
    for g in golden_filtered[:10]:
        print(f"    {g['pattern_key']}: score={g['smart_score']}, rules={g['smart_rules']}")

    # FP которые не были бы отфильтрованы
    fp_not_filtered = data['fp_not_filtered']
    print(f"\n  FP со score >= 60 (не отфильтрованы): {len(fp_not_filtered)}")
    for f in fp_not_filtered[:10]:
        print(f"    {f['pattern_key']}: score={f['smart_score']}, cat={f['category']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Пересчёт SmartFilter данных для БД'
    )
    parser.add_argument('--stats', action='store_true',
                        help='Показать статистику')
    parser.add_argument('--calibrate', action='store_true',
                        help='Показать данные для калибровки')
    parser.add_argument('--no-semantics', action='store_true',
                        help='Не загружать Word2Vec модель')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Минимальный вывод')

    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.calibrate:
        show_calibration()
    else:
        rebuild_all(
            use_semantics=not args.no_semantics,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
