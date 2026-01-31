"""
Менеджер зависимостей фильтрации v1.1.

Централизованная проверка и загрузка всех опциональных зависимостей.
Заменяет 8 блоков try-except в engine.py.

ВАЖНО: Запускать через venv с установленным пакетом (pip install -e .)

v1.2 (2026-01-31): Исправлен импорт ml_classifier через sys.path
v1.1 (2026-01-31): Относительный импорт из Инструменты
v1.0 (2026-01-31): Начальная версия
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable

VERSION = '1.2.0'


@dataclass
class DependencyStatus:
    """Статус загрузки зависимости."""
    name: str
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None
    module: Optional[Any] = None


class Dependencies:
    """
    Менеджер зависимостей фильтрации.

    Централизует загрузку и проверку всех опциональных модулей.

    Использование:
        deps = Dependencies()
        deps.load_all()

        if deps.has_pymorphy:
            lemma = deps.morph.parse(word)[0].normal_form
    """

    def __init__(self):
        # Статусы зависимостей
        self._status: dict[str, DependencyStatus] = {}

        # Загруженные модули/объекты
        self.morph: Optional[Any] = None
        self.scoring_engine: Optional[Any] = None
        self.semantic_manager: Optional[Any] = None
        self.smart_scorer: Optional[Any] = None
        self.smart_filter: Optional[Any] = None
        self.context_verifier: Optional[Any] = None
        self.ml_classifier: Optional[Any] = None
        self.rules_module: Optional[Any] = None

        # Функции из модулей
        self.get_similarity: Callable[[str, str], float] = lambda w1, w2: 0.0
        self.is_hard_negative: Callable[[str, str], bool] = lambda w1, w2: False

    # =========================================================================
    # Флаги доступности (свойства)
    # =========================================================================

    @property
    def has_pymorphy(self) -> bool:
        return self._status.get('pymorphy', DependencyStatus('pymorphy', False)).available

    @property
    def has_scoring_engine(self) -> bool:
        return self._status.get('scoring_engine', DependencyStatus('scoring_engine', False)).available

    @property
    def has_semantic_manager(self) -> bool:
        return self._status.get('semantic_manager', DependencyStatus('semantic_manager', False)).available

    @property
    def has_smart_scorer(self) -> bool:
        return self._status.get('smart_scorer', DependencyStatus('smart_scorer', False)).available

    @property
    def has_smart_filter(self) -> bool:
        return self._status.get('smart_filter', DependencyStatus('smart_filter', False)).available

    @property
    def has_context_verifier(self) -> bool:
        return self._status.get('context_verifier', DependencyStatus('context_verifier', False)).available

    @property
    def has_ml_classifier(self) -> bool:
        return self._status.get('ml_classifier', DependencyStatus('ml_classifier', False)).available

    @property
    def has_rules_module(self) -> bool:
        return self._status.get('rules_module', DependencyStatus('rules_module', False)).available

    # =========================================================================
    # Загрузка зависимостей
    # =========================================================================

    def load_all(self) -> 'Dependencies':
        """Загружает все зависимости. Возвращает self для цепочки."""
        self._load_pymorphy()
        self._load_scoring_engine()
        self._load_semantic_manager()
        self._load_smart_scorer()
        self._load_smart_filter()
        self._load_context_verifier()
        self._load_ml_classifier()
        self._load_rules_module()
        return self

    def _load_pymorphy(self) -> None:
        """Загружает pymorphy3."""
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
            self._status['pymorphy'] = DependencyStatus(
                'pymorphy', True, version=getattr(pymorphy3, '__version__', 'unknown')
            )
        except ImportError as e:
            self._status['pymorphy'] = DependencyStatus('pymorphy', False, error=str(e))

    def _load_scoring_engine(self) -> None:
        """Загружает scoring_engine."""
        try:
            from .scoring_engine import (
                should_filter_by_score, is_hard_negative, HARD_NEGATIVES
            )
            self.scoring_engine = True
            self.is_hard_negative = is_hard_negative
            self._status['scoring_engine'] = DependencyStatus('scoring_engine', True)
        except ImportError as e:
            self._status['scoring_engine'] = DependencyStatus('scoring_engine', False, error=str(e))

    def _load_semantic_manager(self) -> None:
        """Загружает semantic_manager."""
        try:
            from .semantic_manager import get_semantic_manager, get_similarity
            self.semantic_manager = get_semantic_manager
            self.get_similarity = get_similarity
            self._status['semantic_manager'] = DependencyStatus('semantic_manager', True)
        except ImportError as e:
            self._status['semantic_manager'] = DependencyStatus('semantic_manager', False, error=str(e))

    def _load_smart_scorer(self) -> None:
        """Загружает smart_scorer."""
        try:
            from .smart_scorer import SmartScorer, ScoreResult, WEIGHTS as SCORER_WEIGHTS
            self.smart_scorer = SmartScorer
            self._status['smart_scorer'] = DependencyStatus('smart_scorer', True)
        except ImportError as e:
            self._status['smart_scorer'] = DependencyStatus('smart_scorer', False, error=str(e))

    def _load_smart_filter(self) -> None:
        """Загружает smart_filter."""
        try:
            from .smart_filter import SmartFilter, get_smart_filter, evaluate_error_smart
            self.smart_filter = SmartFilter
            self._status['smart_filter'] = DependencyStatus('smart_filter', True)
        except ImportError as e:
            self._status['smart_filter'] = DependencyStatus('smart_filter', False, error=str(e))

    def _load_context_verifier(self) -> None:
        """Загружает context_verifier."""
        try:
            from .context_verifier import should_filter_by_context, VERSION as CV_VERSION
            self.context_verifier = should_filter_by_context
            self._status['context_verifier'] = DependencyStatus(
                'context_verifier', True, version=CV_VERSION
            )
        except ImportError as e:
            self._status['context_verifier'] = DependencyStatus('context_verifier', False, error=str(e))

    def _load_ml_classifier(self) -> None:
        """Загружает ML-классификатор."""
        try:
            # v1.2: Импорт через sys.path
            import sys
            from pathlib import Path
            _script_dir = Path(__file__).parent.parent
            if str(_script_dir) not in sys.path:
                sys.path.insert(0, str(_script_dir))

            from ml_classifier import get_classifier, FalsePositiveClassifier
            classifier = get_classifier()

            if classifier.model is not None:
                self.ml_classifier = classifier
                self._status['ml_classifier'] = DependencyStatus(
                    'ml_classifier', True,
                    version=getattr(classifier, 'VERSION', 'unknown')
                )
            else:
                self._status['ml_classifier'] = DependencyStatus(
                    'ml_classifier', False, error='Model not loaded'
                )
        except Exception as e:
            self._status['ml_classifier'] = DependencyStatus('ml_classifier', False, error=str(e))

    def _load_rules_module(self) -> None:
        """Загружает модуль rules/."""
        try:
            from .rules import (
                apply_protection_layers,
                check_yandex_phonetic_pair,
                check_i_ya_confusion,
                check_alignment_artifact,
                check_safe_ending_transition,
                check_single_consonant_artifact,
                YANDEX_PHONETIC_PAIRS,
                SAFE_ENDING_TRANSITIONS,
                COMPOUND_PARTICLES,
                SINGLE_CONSONANT_ARTIFACTS,
            )
            self.rules_module = True
            self._status['rules_module'] = DependencyStatus('rules_module', True)
        except ImportError as e:
            self._status['rules_module'] = DependencyStatus('rules_module', False, error=str(e))

    # =========================================================================
    # Отчёт о зависимостях
    # =========================================================================

    def get_status_report(self) -> str:
        """Возвращает текстовый отчёт о статусе всех зависимостей."""
        lines = ["Dependencies Status:", "=" * 40]

        for name, status in sorted(self._status.items()):
            icon = "✓" if status.available else "✗"
            version_str = f" v{status.version}" if status.version else ""
            error_str = f" ({status.error})" if status.error else ""
            lines.append(f"  {icon} {name}{version_str}{error_str}")

        return "\n".join(lines)

    def get_available_count(self) -> tuple[int, int]:
        """Возвращает (доступных, всего) зависимостей."""
        available = sum(1 for s in self._status.values() if s.available)
        total = len(self._status)
        return available, total


# Глобальный экземпляр
_deps: Optional[Dependencies] = None


def get_dependencies() -> Dependencies:
    """
    Возвращает глобальный экземпляр Dependencies.

    При первом вызове загружает все зависимости и инициализирует флаги совместимости.
    """
    global _deps
    if _deps is None:
        _deps = Dependencies()
        _deps.load_all()
        _init_compat_flags_from_deps(_deps)
    return _deps


def reload_dependencies() -> Dependencies:
    """Перезагружает все зависимости."""
    global _deps
    _deps = Dependencies()
    _deps.load_all()
    _init_compat_flags_from_deps(_deps)
    return _deps


# =========================================================================
# Совместимость с существующим кодом engine.py
# =========================================================================

# Эти переменные будут установлены при первой загрузке
HAS_PYMORPHY = False
HAS_SCORING_ENGINE = False
HAS_SEMANTIC_MANAGER = False
HAS_SMART_SCORER = False
HAS_SMART_FILTER = False
HAS_CONTEXT_VERIFIER = False
HAS_ML_CLASSIFIER = False
HAS_RULES_MODULE = False


def _init_compat_flags_from_deps(deps: Dependencies) -> None:
    """Инициализирует флаги совместимости из загруженных зависимостей."""
    global HAS_PYMORPHY, HAS_SCORING_ENGINE, HAS_SEMANTIC_MANAGER
    global HAS_SMART_SCORER, HAS_SMART_FILTER, HAS_CONTEXT_VERIFIER
    global HAS_ML_CLASSIFIER, HAS_RULES_MODULE

    HAS_PYMORPHY = deps.has_pymorphy
    HAS_SCORING_ENGINE = deps.has_scoring_engine
    HAS_SEMANTIC_MANAGER = deps.has_semantic_manager
    HAS_SMART_SCORER = deps.has_smart_scorer
    HAS_SMART_FILTER = deps.has_smart_filter
    HAS_CONTEXT_VERIFIER = deps.has_context_verifier
    HAS_ML_CLASSIFIER = deps.has_ml_classifier
    HAS_RULES_MODULE = deps.has_rules_module


if __name__ == '__main__':
    deps = Dependencies()
    deps.load_all()
    print(f"Dependencies v{VERSION}")
    print(deps.get_status_report())
    available, total = deps.get_available_count()
    print(f"\nLoaded: {available}/{total}")
