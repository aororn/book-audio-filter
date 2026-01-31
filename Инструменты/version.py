"""
Единый источник версий проекта Яндекс Спич.

Все версии модулей и проекта определены здесь.
Другие модули должны импортировать версии отсюда.

v1.0.0 (2026-01-30): Начальная версия
"""

# =============================================================================
# ВЕРСИЯ ПРОЕКТА
# =============================================================================

PROJECT_VERSION = '14.1.0'
PROJECT_DATE = '2026-01-31'

# =============================================================================
# ВЕРСИИ МОДУЛЕЙ (синхронизированы с реальными версиями в файлах)
# =============================================================================

# Фильтрация
FILTER_ENGINE_VERSION = '9.7.0'       # v9.7.0: Исправлена статистика (2026-01-31)
CONTEXT_VERIFIER_VERSION = '4.1.0'
RULES_MODULE_VERSION = '1.0.0'
MORPHO_RULES_VERSION = '1.2.0'        # Реальная версия в morpho_rules.py
COMPARISON_VERSION = '6.2.0'           # Реальная версия в comparison.py

# Выравнивание
SMART_COMPARE_VERSION = '10.6.0'       # Реальная версия в smart_compare.py
ALIGNMENT_MANAGER_VERSION = '1.2.0'

# ML и семантика
ML_CLASSIFIER_VERSION = '2.0.0'        # v2.0.0: Контекстные признаки (31 признак)
SEMANTIC_MANAGER_VERSION = '2.0.0'

# Тестирование
TEST_RUNNER_VERSION = '6.3.0'

# Пакет фильтров
FILTERS_PACKAGE_VERSION = '8.2.0'     # Реальная версия в __init__.py

# =============================================================================
# ЗАВИСИМОСТИ ВЕРСИЙ (для валидации)
# =============================================================================

# Минимальная версия smart_compare для корректной работы фильтра
MIN_SMART_COMPARE_FOR_FILTER = '10.5.0'

# =============================================================================
# УТИЛИТЫ
# =============================================================================

def parse_version(v: str) -> tuple:
    """Парсит версию в кортеж для сравнения."""
    try:
        parts = v.split('.')
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def is_version_compatible(current: str, minimum: str) -> bool:
    """Проверяет совместимость версий."""
    return parse_version(current) >= parse_version(minimum)


def get_version_string() -> str:
    """Возвращает полную строку версии проекта."""
    return f"YandexSpich v{PROJECT_VERSION} (filter={FILTER_ENGINE_VERSION}, compare={SMART_COMPARE_VERSION})"


def get_version_info() -> dict:
    """Возвращает словарь с информацией о версиях."""
    return {
        'project': PROJECT_VERSION,
        'project_date': PROJECT_DATE,
        'filter_engine': FILTER_ENGINE_VERSION,
        'context_verifier': CONTEXT_VERIFIER_VERSION,
        'smart_compare': SMART_COMPARE_VERSION,
        'morpho_rules': MORPHO_RULES_VERSION,
        'comparison': COMPARISON_VERSION,
        'ml_classifier': ML_CLASSIFIER_VERSION,
        'semantic_manager': SEMANTIC_MANAGER_VERSION,
        'filters_package': FILTERS_PACKAGE_VERSION,
        'test_runner': TEST_RUNNER_VERSION,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print(get_version_string())
    print()
    print("Версии модулей:")
    for name, version in get_version_info().items():
        print(f"  {name}: {version}")
