.PHONY: test lint lint-fix typecheck coverage check-chapter-1 check-chapter-2 check-all clean help

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# ==============================================================================
# Тесты
# ==============================================================================

test:  ## Запустить все unit-тесты
	$(PYTEST) -v

test-fast:  ## Запустить тесты без медленных
	$(PYTEST) -v -m "not slow"

test-coverage:  ## Запустить тесты с отчётом покрытия
	$(PYTEST) --cov=Инструменты --cov-report=html --cov-report=term-missing

# ==============================================================================
# Линтинг и проверка типов
# ==============================================================================

lint:  ## Проверить код линтером (ruff)
	$(RUFF) check Инструменты/ Тесты/

lint-fix:  ## Автоматически исправить ошибки линтера
	$(RUFF) check --fix Инструменты/ Тесты/

typecheck:  ## Проверка типов (mypy)
	$(MYPY) Инструменты/

format:  ## Форматировать код (ruff)
	$(RUFF) format Инструменты/ Тесты/

# ==============================================================================
# Проверка глав
# ==============================================================================

check-chapter-1:  ## Проверить главу 1 (gold standard)
	$(PYTHON) Тесты/run_full_test.py --chapter 1

check-chapter-2:  ## Проверить главу 2 (gold standard)
	$(PYTHON) Тесты/run_full_test.py --chapter 2

check-all:  ## Проверить все главы (gold standard)
	$(PYTHON) Тесты/run_full_test.py

# ==============================================================================
# Пакетная обработка
# ==============================================================================

batch:  ## Пакетная обработка всех глав (последовательно)
	$(PYTHON) Инструменты/batch_pipeline.py -a Оригинал/Аудио -t Оригинал/Главы

batch-parallel:  ## Пакетная обработка всех глав (параллельно, 3 процесса)
	$(PYTHON) Инструменты/batch_pipeline.py -a Оригинал/Аудио -t Оригинал/Главы --parallel --workers 3

# ==============================================================================
# Утилиты
# ==============================================================================

clean:  ## Очистить временные файлы
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

help:  ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
