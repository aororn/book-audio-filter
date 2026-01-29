#!/usr/bin/env python3
"""
conftest.py — общие фикстуры и конфигурация для pytest

Настройка путей, общие данные для тестов.
"""

import sys
from pathlib import Path

import pytest


# =============================================================================
# НАСТРОЙКА ПУТЕЙ
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
INSTRUMENTS_DIR = PROJECT_DIR / 'Инструменты'
TESTS_DIR = PROJECT_DIR / 'Тесты'

# Добавляем путь к модулям один раз
if str(INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(INSTRUMENTS_DIR))


# =============================================================================
# МАРКЕРЫ
# =============================================================================

def pytest_configure(config):
    """Регистрируем кастомные маркеры"""
    config.addinivalue_line("markers", "slow: тесты, требующие загрузки данных или API")
    config.addinivalue_line("markers", "pymorphy: тесты, требующие pymorphy2")
    config.addinivalue_line("markers", "integration: интеграционные тесты")


# Исключаем утилитарные скрипты из обнаружения тестов
collect_ignore = [
    str(TESTS_DIR / 'test_golden_standard.py'),
    str(TESTS_DIR / 'run_full_test.py'),
]


# =============================================================================
# ОБЩИЕ ФИКСТУРЫ
# =============================================================================

@pytest.fixture
def project_dir():
    """Путь к корню проекта"""
    return PROJECT_DIR


@pytest.fixture
def tests_dir():
    """Путь к папке тестов"""
    return TESTS_DIR


@pytest.fixture
def instruments_dir():
    """Путь к папке инструментов"""
    return INSTRUMENTS_DIR


# =============================================================================
# ФИКСТУРЫ ОШИБОК
# =============================================================================

@pytest.fixture
def substitution_error():
    """Типичная ошибка подстановки"""
    return {
        'type': 'substitution',
        'wrong': 'живем',
        'correct': 'живы',
        'time': 120.5,
    }


@pytest.fixture
def deletion_error():
    """Типичная ошибка удаления"""
    return {
        'type': 'deletion',
        'correct': 'слово',
        'time': 45.0,
    }


@pytest.fixture
def insertion_error():
    """Типичная ошибка вставки"""
    return {
        'type': 'insertion',
        'wrong': 'лишнее',
        'time': 200.0,
    }


@pytest.fixture
def homophone_error():
    """Ошибка-омофон (должна фильтроваться)"""
    return {
        'type': 'substitution',
        'wrong': 'ну',
        'correct': 'но',
    }


@pytest.fixture
def yandex_typical_error():
    """Типичная ошибка Яндекса (должна фильтроваться)"""
    return {
        'type': 'substitution',
        'wrong': 'сто',
        'correct': 'то',
    }


@pytest.fixture
def real_reader_error():
    """Реальная ошибка чтеца (НЕ должна фильтроваться)"""
    return {
        'type': 'substitution',
        'wrong': 'выхода',
        'correct': 'способа',
    }


# =============================================================================
# ФИКСТУРЫ СЛОВ
# =============================================================================

@pytest.fixture
def sample_word_pairs():
    """Пары слов для тестирования фильтров"""
    return {
        'homophones': [
            ('его', 'ево'),
            ('что', 'што'),
            ('ну', 'но'),
        ],
        'grammar': [
            ('красный', 'красное'),
            ('заметит', 'заметят'),
        ],
        'different': [
            ('дом', 'кот'),
            ('слово', 'книга'),
        ],
    }
