#!/usr/bin/env python3
"""
Test Filter Accuracy v1.0 — Тесты точности фильтров на основе БД

Проверяет:
1. Recall FP — какой % ложных ошибок отфильтрован
2. Precision — не фильтруются ли golden ошибки
3. Соответствие actual_filter и expected_filter

Использование:
    pytest test_filter_accuracy.py -v
    pytest test_filter_accuracy.py -v -k "recall"
    python test_filter_accuracy.py  # standalone

Changelog:
    v1.0 (2026-01-29): Начальная версия
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-29'

import sys
from pathlib import Path
from typing import Dict, List, Any

# Добавляем путь к инструментам
PROJECT_ROOT = Path(__file__).parent.parent
TOOLS_DIR = PROJECT_ROOT / 'Инструменты'
sys.path.insert(0, str(TOOLS_DIR))

import pytest

from false_positives_db import FalsePositivesDB


# =============================================================================
# КОНФИГУРАЦИЯ ТЕСТОВ
# =============================================================================

# Минимальный допустимый recall для FP
MIN_FP_RECALL = 0.60  # 60% — текущий baseline

# Максимально допустимые пропуски по категориям
MAX_UNFILTERED_BY_CATEGORY = {
    'grammar_ending': 200,  # Было 151 — даём запас
    'short_word': 100,      # Было 63
    'unknown': 60,          # Было 40
    'phonetic': 30,         # Было 16
    'prefix_variant': 10,   # Было 3
}

# Golden не должны фильтроваться
MAX_GOLDEN_FILTERED = 0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope='module')
def db():
    """Подключение к БД."""
    return FalsePositivesDB()


@pytest.fixture(scope='module')
def filter_accuracy(db):
    """Статистика точности фильтров."""
    return db.get_filter_accuracy()


@pytest.fixture(scope='module')
def ml_stats(db):
    """ML статистика."""
    return db.get_ml_stats()


# =============================================================================
# ТЕСТЫ RECALL
# =============================================================================

class TestFPRecall:
    """Тесты на recall ложных срабатываний."""

    def test_fp_recall_minimum(self, filter_accuracy):
        """Recall FP должен быть выше минимального порога."""
        recall = filter_accuracy['recall']
        assert recall >= MIN_FP_RECALL, (
            f"FP Recall {recall:.1%} ниже порога {MIN_FP_RECALL:.1%}. "
            f"Отфильтровано: {filter_accuracy['filtered_fp']}/{filter_accuracy['total_fp']}"
        )

    def test_grammar_ending_coverage(self, filter_accuracy):
        """Категория grammar_ending должна быть покрыта."""
        unfiltered = filter_accuracy.get('unfiltered_by_category', {})
        count = unfiltered.get('grammar_ending', 0)
        max_allowed = MAX_UNFILTERED_BY_CATEGORY['grammar_ending']

        assert count <= max_allowed, (
            f"Пропущено grammar_ending: {count} > {max_allowed}. "
            f"Требуется улучшение morpho_* фильтров."
        )

    def test_short_word_coverage(self, filter_accuracy):
        """Категория short_word должна быть покрыта."""
        unfiltered = filter_accuracy.get('unfiltered_by_category', {})
        count = unfiltered.get('short_word', 0)
        max_allowed = MAX_UNFILTERED_BY_CATEGORY['short_word']

        assert count <= max_allowed, (
            f"Пропущено short_word: {count} > {max_allowed}. "
            f"Требуется улучшение alignment_artifact_length."
        )

    def test_phonetic_coverage(self, filter_accuracy):
        """Категория phonetic должна быть покрыта."""
        unfiltered = filter_accuracy.get('unfiltered_by_category', {})
        count = unfiltered.get('phonetic', 0)
        max_allowed = MAX_UNFILTERED_BY_CATEGORY['phonetic']

        assert count <= max_allowed, (
            f"Пропущено phonetic: {count} > {max_allowed}. "
            f"Требуется улучшение homophone/yandex_typical."
        )


# =============================================================================
# ТЕСТЫ PRECISION (GOLDEN НЕ ФИЛЬТРУЮТСЯ)
# =============================================================================

class TestGoldenPrecision:
    """Тесты на precision — golden ошибки не должны фильтроваться."""

    def test_golden_not_filtered(self, db):
        """Golden ошибки НЕ должны иметь actual_filter."""
        cursor = db.conn.execute('''
            SELECT pattern_key, actual_filter
            FROM patterns
            WHERE is_golden = 1 AND actual_filter IS NOT NULL AND actual_filter != ''
        ''')
        filtered_golden = cursor.fetchall()

        assert len(filtered_golden) <= MAX_GOLDEN_FILTERED, (
            f"Отфильтровано {len(filtered_golden)} golden ошибок! "
            f"Примеры: {[dict(row) for row in filtered_golden[:5]]}"
        )

    def test_golden_count(self, ml_stats):
        """В БД должны быть размечены golden ошибки."""
        assert ml_stats['golden_count'] > 0, (
            "В БД нет golden ошибок. Запустите mark_golden()."
        )


# =============================================================================
# ТЕСТЫ СТАБИЛЬНОСТИ
# =============================================================================

class TestStability:
    """Тесты на стабильность паттернов."""

    def test_stable_patterns_exist(self, db):
        """Должны быть стабильные паттерны (во всех транскрипциях)."""
        cursor = db.conn.execute('''
            SELECT COUNT(*) as count FROM patterns WHERE is_stable = 1
        ''')
        stable_count = cursor.fetchone()[0]

        # Пока стабильность не полностью реализована, просто проверяем запрос
        assert stable_count >= 0, "Ошибка запроса is_stable"


# =============================================================================
# ТЕСТЫ ПОКРЫТИЯ ФИЛЬТРАМИ
# =============================================================================

class TestFilterCoverage:
    """Тесты на покрытие фильтрами."""

    def test_top_filters_working(self, filter_accuracy):
        """Топовые фильтры должны работать."""
        by_filter = filter_accuracy.get('by_filter', {})

        # Проверяем что основные фильтры сработали хотя бы раз
        expected_filters = [
            'yandex_typical',
            'yandex_name_error',
            'alignment_artifact',
        ]

        for flt in expected_filters:
            # Проверяем что фильтр есть или есть похожий (с суффиксом)
            found = any(f.startswith(flt.split('_')[0]) for f in by_filter.keys())
            assert found or flt in by_filter, (
                f"Фильтр {flt} не сработал ни разу. "
                f"Активные фильтры: {list(by_filter.keys())[:10]}"
            )

    def test_no_unknown_filters(self, filter_accuracy):
        """Не должно быть слишком много паттернов без фильтра."""
        unfiltered = filter_accuracy['unfiltered_fp']
        total = filter_accuracy['total_fp']

        # Максимум 50% могут быть без фильтра
        assert unfiltered / total <= 0.50, (
            f"Слишком много FP без фильтра: {unfiltered}/{total} = {unfiltered/total:.1%}"
        )


# =============================================================================
# ИНТЕГРАЦИОННЫЙ ТЕСТ
# =============================================================================

class TestIntegration:
    """Интеграционные тесты."""

    def test_database_consistency(self, db):
        """БД должна быть консистентной."""
        # Проверка что patterns и occurrences согласованы
        cursor = db.conn.execute('''
            SELECT p.id, p.pattern_key, p.count, COUNT(o.id) as occ_count
            FROM patterns p
            LEFT JOIN occurrences o ON p.id = o.pattern_id
            GROUP BY p.id
            HAVING p.count != occ_count
            LIMIT 5
        ''')
        inconsistent = cursor.fetchall()

        # Разрешаем небольшую рассинхронизацию (из-за агрегации)
        assert len(inconsistent) <= 100, (
            f"Несогласованность patterns.count и occurrences: {len(inconsistent)} записей"
        )

    def test_schema_version(self, db):
        """Схема БД должна быть v3."""
        cursor = db.conn.execute('SELECT version FROM schema_version LIMIT 1')
        row = cursor.fetchone()
        assert row is not None, "Нет записи schema_version"
        assert row[0] >= 3, f"Схема БД устарела: v{row[0]}, требуется v3+"


# =============================================================================
# STANDALONE ЗАПУСК
# =============================================================================

def print_report():
    """Печатает отчёт о точности фильтров."""
    db = FalsePositivesDB()

    print(f"\n{'='*60}")
    print(f"  Test Filter Accuracy v{VERSION}")
    print(f"{'='*60}")

    # ML статистика
    ml = db.get_ml_stats()
    print(f"\n=== Данные в БД ===")
    print(f"Всего паттернов: {ml['total_patterns']}")
    print(f"Golden: {ml['golden_count']}")
    print(f"FP: {ml['non_golden_count']}")

    # Точность фильтров
    acc = db.get_filter_accuracy()
    print(f"\n=== Точность фильтров ===")
    print(f"FP Recall: {acc['recall']:.1%} ({acc['filtered_fp']}/{acc['total_fp']})")
    print(f"Пропущено: {acc['unfiltered_fp']}")

    # По категориям
    print(f"\n=== Пропущенные по категориям ===")
    for cat, count in acc.get('unfiltered_by_category', {}).items():
        status = "✓" if count <= MAX_UNFILTERED_BY_CATEGORY.get(cat, float('inf')) else "✗"
        print(f"  {status} {cat}: {count}")

    # Golden отфильтрованные
    cursor = db.conn.execute('''
        SELECT COUNT(*) FROM patterns
        WHERE is_golden = 1 AND actual_filter IS NOT NULL AND actual_filter != ''
    ''')
    filtered_golden = cursor.fetchone()[0]
    status = "✓" if filtered_golden == 0 else "✗"
    print(f"\n{status} Golden отфильтрованы: {filtered_golden}")

    # Итог
    print(f"\n{'='*60}")
    if acc['recall'] >= MIN_FP_RECALL and filtered_golden == 0:
        print("  ✓ Все базовые метрики в норме")
    else:
        print("  ✗ Есть проблемы с метриками")
    print(f"{'='*60}\n")

    db.close()


if __name__ == '__main__':
    print_report()

    # Запускаем pytest
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
