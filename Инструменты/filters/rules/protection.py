"""
Защитные слои фильтрации v1.1.

Эти правила выполняются ПЕРВЫМИ и защищают реальные ошибки от фильтрации.
Если защитный слой срабатывает — ошибка НЕ фильтруется.

Слои:
1. HARD_NEGATIVES — известные пары путаницы (scoring_engine.py)
2. SEMANTIC_SLIP — оговорки чтеца (semantic_manager.py)

v1.1 (2026-01-31): Пороги из config.py (без дублирования)
v1.0 (2026-01-30): Извлечено из engine.py v8.9
"""

from typing import Tuple, Optional

# v1.1: Пороги теперь берутся из config.py
try:
    from ..config import get_semantic_slip_threshold, get_phonetic_slip_threshold
except ImportError:
    # Fallback для изолированного запуска
    def get_semantic_slip_threshold() -> float:
        return 0.4
    def get_phonetic_slip_threshold() -> float:
        return 0.7


def check_hard_negatives(w1: str, w2: str) -> Tuple[bool, str]:
    """
    Проверяет HARD_NEGATIVES — известные пары путаницы.

    Если пара в HARD_NEGATIVES — это реальная ошибка, НЕ фильтруем.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)

    Returns:
        (should_protect, reason) — если should_protect=True, НЕ фильтровать
    """
    try:
        from ..scoring_engine import is_hard_negative
        if is_hard_negative(w1, w2):
            return True, 'PROTECTED_hard_negative'
    except ImportError:
        pass

    return False, ''


def check_semantic_slip(
    w1: str,
    w2: str,
    threshold: float = None
) -> Tuple[bool, str]:
    """
    Проверяет семантическую близость для детекции оговорок.

    Высокая семантическая близость + разные леммы = оговорка чтеца.
    Оговорки — реальные ошибки, НЕ фильтруем.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        threshold: Порог семантической близости

    Returns:
        (should_protect, reason) — если should_protect=True, НЕ фильтровать
    """
    # v1.1: Порог из config.py если не передан явно
    if threshold is None:
        threshold = get_semantic_slip_threshold()

    try:
        from ..semantic_manager import get_similarity
        from ..comparison import get_lemma, HAS_PYMORPHY

        if not HAS_PYMORPHY:
            return False, ''

        lemma1 = get_lemma(w1)
        lemma2 = get_lemma(w2)

        # Только для разных лемм — проверяем семантику
        if lemma1 and lemma2 and lemma1 != lemma2:
            semantic_sim = get_similarity(w1, w2)
            if semantic_sim >= threshold:
                return True, f'PROTECTED_semantic_slip({semantic_sim:.2f})'

    except ImportError:
        pass

    return False, ''


def apply_protection_layers(
    w1: str,
    w2: str,
    error_type: str = 'substitution'
) -> Tuple[bool, str]:
    """
    Применяет все защитные слои последовательно.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        error_type: Тип ошибки

    Returns:
        (should_protect, reason) — если should_protect=True, НЕ фильтровать
    """
    if error_type != 'substitution':
        return False, ''

    # Слой 1: HARD_NEGATIVES
    protected, reason = check_hard_negatives(w1, w2)
    if protected:
        return True, reason

    # Слой 2: Semantic slip
    protected, reason = check_semantic_slip(w1, w2)
    if protected:
        return True, reason

    return False, ''
