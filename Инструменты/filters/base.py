"""
Base classes for filter rules - ABC interface for extensible filtering

Этот модуль определяет базовый интерфейс для правил фильтрации,
позволяя легко добавлять новые правила без изменения engine.py.

Использование:
    from filters.base import FilterRule, register_rule

    class MyCustomRule(FilterRule):
        name = "my_custom_rule"
        priority = 15  # 0-19, меньше = раньше выполняется

        def should_filter(self, error, context):
            if some_condition(error):
                return True, "my_custom_rule"
            return False, ""

    # Регистрируем правило
    register_rule(MyCustomRule())

Changelog:
    v1.0 (2026-01-25): Начальная версия
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FilterContext:
    """
    Контекст для фильтрации — все данные, доступные правилу.

    Attributes:
        all_errors: Список всех ошибок (для детектора цепочек)
        error_index: Индекс текущей ошибки в списке
        config: Конфигурация фильтра
        character_names: Множество имён персонажей
        protected_words: Множество защищённых слов
    """
    all_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_index: int = 0
    config: Optional[Any] = None
    character_names: set = field(default_factory=set)
    protected_words: set = field(default_factory=set)


class FilterRule(ABC):
    """
    Абстрактный базовый класс для правил фильтрации.

    Каждое правило определяет:
    - name: уникальное имя для логирования
    - priority: порядок выполнения (0-19, меньше = раньше)
    - error_types: типы ошибок, к которым применяется ('all' или список)
    - should_filter(): логика фильтрации

    Example:
        class HomophoneRule(FilterRule):
            name = "homophone"
            priority = 5
            error_types = ['substitution']

            def should_filter(self, error, context):
                from .comparison import is_homophone_match
                wrong = error.get('wrong', '')
                correct = error.get('correct', '')
                if is_homophone_match(wrong, correct):
                    return True, "homophone"
                return False, ""
    """

    # Имя правила (для логирования и отладки)
    name: str = "base_rule"

    # Приоритет выполнения: 0-19
    # Меньше = раньше выполняется
    # Рекомендации:
    #   0-4: Артефакты выравнивания, защищённые слова
    #   5-9: Омофоны, грамматика
    #   10-14: Морфология, леммы
    #   15-19: Цепочки, контекстные правила
    priority: int = 10

    # Типы ошибок, к которым применяется правило
    # 'all' — ко всем типам
    # ['substitution', 'insertion'] — только к указанным
    error_types: Union[str, List[str]] = 'all'

    @abstractmethod
    def should_filter(
        self,
        error: Dict[str, Any],
        context: FilterContext
    ) -> Tuple[bool, str]:
        """
        Определяет, нужно ли отфильтровать ошибку.

        Args:
            error: Словарь с данными ошибки (type, wrong, correct, time, context, ...)
            context: Контекст фильтрации (все ошибки, конфиг, словари)

        Returns:
            Tuple[bool, str]: (True если отфильтровать, причина фильтрации)
            Если не фильтруем, возвращаем (False, "")
        """
        pass

    def applies_to(self, error_type: str) -> bool:
        """Проверяет, применяется ли правило к данному типу ошибки"""
        if self.error_types == 'all':
            return True
        if isinstance(self.error_types, list):
            return error_type in self.error_types
        return self.error_types == error_type


# =============================================================================
# РЕЕСТР ПРАВИЛ
# =============================================================================

_rules_registry: List[FilterRule] = []


def register_rule(rule: FilterRule) -> None:
    """
    Регистрирует правило в глобальном реестре.

    Args:
        rule: Экземпляр FilterRule
    """
    _rules_registry.append(rule)
    # Сортируем по приоритету
    _rules_registry.sort(key=lambda r: r.priority)


def unregister_rule(rule_name: str) -> bool:
    """
    Удаляет правило из реестра по имени.

    Args:
        rule_name: Имя правила

    Returns:
        True если правило было найдено и удалено
    """
    global _rules_registry
    initial_len = len(_rules_registry)
    _rules_registry = [r for r in _rules_registry if r.name != rule_name]
    return len(_rules_registry) < initial_len


def get_registered_rules() -> List[FilterRule]:
    """Возвращает список зарегистрированных правил (отсортированный по приоритету)"""
    return _rules_registry.copy()


def clear_rules_registry() -> None:
    """Очищает реестр правил"""
    global _rules_registry
    _rules_registry = []


def apply_registered_rules(
    error: Dict[str, Any],
    context: FilterContext
) -> Tuple[bool, str]:
    """
    Применяет все зарегистрированные правила к ошибке.

    Args:
        error: Словарь с данными ошибки
        context: Контекст фильтрации

    Returns:
        Tuple[bool, str]: (True если отфильтровано, причина)
    """
    error_type = error.get('type', 'substitution')

    for rule in _rules_registry:
        if rule.applies_to(error_type):
            should_filter, reason = rule.should_filter(error, context)
            if should_filter:
                return True, reason

    return False, ""


# =============================================================================
# ПРИМЕР ПРАВИЛ
# =============================================================================

class ExampleHomophoneRule(FilterRule):
    """
    Пример реализации правила для омофонов.

    Это демонстрационный класс — реальная логика в comparison.py.
    """
    name = "example_homophone"
    priority = 5
    error_types = ['substitution']

    def should_filter(self, error: Dict[str, Any], context: FilterContext) -> Tuple[bool, str]:
        # Импортируем реальную функцию
        from .comparison import is_homophone_match

        wrong = error.get('wrong', '')
        correct = error.get('correct', '')

        if is_homophone_match(wrong, correct):
            return True, "homophone"

        return False, ""


class ExampleGrammarRule(FilterRule):
    """
    Пример правила для грамматических окончаний.
    """
    name = "example_grammar"
    priority = 6
    error_types = ['substitution']

    def should_filter(self, error: Dict[str, Any], context: FilterContext) -> Tuple[bool, str]:
        from .comparison import is_grammar_ending_match

        wrong = error.get('wrong', '')
        correct = error.get('correct', '')

        if is_grammar_ending_match(wrong, correct):
            return True, "grammar_ending"

        return False, ""


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def create_context(
    all_errors: List[Dict[str, Any]],
    error_index: int = 0,
    config: Any = None,
    character_names: set = None,
    protected_words: set = None
) -> FilterContext:
    """
    Создаёт контекст фильтрации.

    Args:
        all_errors: Список всех ошибок
        error_index: Индекс текущей ошибки
        config: Конфигурация фильтра
        character_names: Множество имён персонажей
        protected_words: Множество защищённых слов

    Returns:
        FilterContext для передачи в правила
    """
    return FilterContext(
        all_errors=all_errors or [],
        error_index=error_index,
        config=config,
        character_names=character_names or set(),
        protected_words=protected_words or set(),
    )


# =============================================================================
# ЭКСПОРТ
# =============================================================================

__all__ = [
    # Классы
    'FilterRule',
    'FilterContext',

    # Реестр
    'register_rule',
    'unregister_rule',
    'get_registered_rules',
    'clear_rules_registry',
    'apply_registered_rules',

    # Утилиты
    'create_context',

    # Примеры
    'ExampleHomophoneRule',
    'ExampleGrammarRule',
]
