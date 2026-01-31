"""
Фонетико-семантические фильтры v1.0.

Объединяет фильтры, работающие с фонетикой и семантикой:
- same_phonetic_diff_lemma (ex-уровень 0.4)
- high_phon_sem_diff_lemma (ex-уровень 0.45)
- perfect_phon_diff_lemma (ex-уровень 0.5)

Все эти фильтры используют:
- pymorphy (get_lemma)
- phonetic_normalize
- SemanticManager (get_similarity)

v1.0 (2026-01-31): Объединены из engine.py v9.19
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-31'

from typing import Dict, Any, Tuple, Optional, Set

# Импорт морфологии
try:
    from .comparison import (
        get_lemma, phonetic_normalize,
        HAS_PYMORPHY,
    )
except ImportError:
    HAS_PYMORPHY = False
    get_lemma = None
    phonetic_normalize = lambda x: x.lower()

# Импорт семантики
try:
    from .semantic_manager import get_similarity
    HAS_SEMANTIC_MANAGER = True
except ImportError:
    HAS_SEMANTIC_MANAGER = False
    get_similarity = None


# =============================================================================
# ПОРОГИ
# =============================================================================

# Уровень 0.4: Одинаковая фонетика
MIN_WORD_LENGTH = 3  # Исключаем короткие слова (и, я, а)

# Уровень 0.45: Высокая фонетика + семантика
HIGH_PHON_THRESHOLD = 0.8
HIGH_SEM_THRESHOLD = 0.5

# Уровень 0.5: Идеальная фонетика
PERFECT_PHON_THRESHOLD = 0.99


# =============================================================================
# ЗАЩИТНЫЕ СПИСКИ
# =============================================================================

# Притяжательные местоимения разных лиц — реальные ошибки
PROTECTED_POSSESSIVE_PAIRS: Set[Tuple[str, str]] = {
    ('наш', 'ваш'), ('ваш', 'наш'),
    ('наша', 'ваша'), ('ваша', 'наша'),
    ('наши', 'ваши'), ('ваши', 'наши'),
    ('нашу', 'вашу'), ('вашу', 'нашу'),
    ('нашей', 'вашей'), ('вашей', 'нашей'),
    ('нашего', 'вашего'), ('вашего', 'нашего'),
    ('нашим', 'вашим'), ('вашим', 'нашим'),
    ('нашими', 'вашими'), ('вашими', 'нашими'),
    ('нашем', 'вашем'), ('вашем', 'нашем'),
}

# Пары с разными корнями — реальные ошибки
PROTECTED_DIFFERENT_ROOTS: Set[Tuple[str, str]] = {
    ('образ', 'образец'), ('образец', 'образ'),
}


# =============================================================================
# ФУНКЦИИ ФИЛЬТРАЦИИ
# =============================================================================

def check_same_phonetic_diff_lemma(
    w1: str,
    w2: str,
) -> Tuple[bool, str]:
    """
    Уровень 0.4: Одинаковая фонетика, разные леммы.

    Если слова звучат ОДИНАКОВО, но имеют разные леммы — это ошибка ASR.
    Примеры: устранять→устранить, прочие→прочее, открыта→открыто

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)

    Returns:
        (True, reason) если нужно фильтровать
        (False, '') если нет
    """
    if not HAS_PYMORPHY:
        return False, ''

    # Исключаем короткие слова (служебные: и, я, а)
    if len(w1) < MIN_WORD_LENGTH or len(w2) < MIN_WORD_LENGTH:
        return False, ''

    # Проверяем фонетику
    phon1 = phonetic_normalize(w1)
    phon2 = phonetic_normalize(w2)

    if phon1 != phon2:
        return False, ''

    # Проверяем леммы
    lemma1 = get_lemma(w1)
    lemma2 = get_lemma(w2)

    if lemma1 == lemma2:
        return False, ''

    return True, f'same_phonetic_diff_lemma:{phon1}'


def check_high_phon_sem_diff_lemma(
    w1: str,
    w2: str,
    phon_sim: float,
) -> Tuple[bool, str]:
    """
    Уровень 0.45: Высокая фонетика + семантика + разные леммы.

    Критерии:
    - phon >= 0.8
    - sem >= 0.5
    - разные леммы
    - первая буква разная (артефакт ASR: вглубь→глубь)

    Примеры FP: вглубь→глубь, хотелось→захотелось, молчал→помолчал

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        phon_sim: Фонетическая схожесть (0-1)

    Returns:
        (True, reason) если нужно фильтровать
        (False, '') если нет
    """
    if not HAS_PYMORPHY or not HAS_SEMANTIC_MANAGER:
        return False, ''

    # Проверяем первые буквы — должны быть РАЗНЫЕ
    if not w1 or not w2:
        return False, ''

    if w1[0].lower() == w2[0].lower():
        return False, ''

    # Проверяем леммы
    lemma1 = get_lemma(w1)
    lemma2 = get_lemma(w2)

    if not lemma1 or not lemma2 or lemma1 == lemma2:
        return False, ''

    # Проверяем фонетику
    if phon_sim < HIGH_PHON_THRESHOLD:
        return False, ''

    # Проверяем семантику
    sem_sim = get_similarity(w1, w2)
    if sem_sim < HIGH_SEM_THRESHOLD:
        return False, ''

    # Защита: притяжательные местоимения разных лиц
    if (w1.lower(), w2.lower()) in PROTECTED_POSSESSIVE_PAIRS:
        return False, ''

    return True, f'high_phon_sem_diff_lemma:phon={phon_sim:.2f},sem={sem_sim:.2f}'


def check_perfect_phon_diff_lemma(
    w1: str,
    w2: str,
    phon_sim: float,
) -> Tuple[bool, str]:
    """
    Уровень 0.5: Идеальное фонетическое совпадение + разные леммы.

    Критерии:
    - phon >= 0.99
    - разные леммы
    - любая семантика

    Примеры FP: прочие→прочее, эта→это, обоснованно→обосновано

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        phon_sim: Фонетическая схожесть (0-1)

    Returns:
        (True, reason) если нужно фильтровать
        (False, '') если нет
    """
    if not HAS_PYMORPHY:
        return False, ''

    # Проверяем фонетику — должна быть идеальная
    if phon_sim < PERFECT_PHON_THRESHOLD:
        return False, ''

    # Проверяем леммы
    lemma1 = get_lemma(w1)
    lemma2 = get_lemma(w2)

    if not lemma1 or not lemma2 or lemma1 == lemma2:
        return False, ''

    # Защита: пары с разными корнями
    if (lemma1.lower(), lemma2.lower()) in PROTECTED_DIFFERENT_ROOTS:
        return False, ''

    return True, f'perfect_phon_diff_lemma:phon={phon_sim:.2f}'


def check_phonetic_semantic(
    error: Dict[str, Any],
    words_norm: list,
) -> Tuple[bool, str]:
    """
    Главная функция: проверяет все фонетико-семантические фильтры.

    Порядок проверок:
    1. same_phonetic_diff_lemma (0.4)
    2. high_phon_sem_diff_lemma (0.45)
    3. perfect_phon_diff_lemma (0.5)

    Args:
        error: Словарь с ошибкой
        words_norm: Нормализованные слова [wrong, correct]

    Returns:
        (True, reason) если нужно фильтровать
        (False, '') если нет
    """
    if error.get('type') != 'substitution':
        return False, ''

    if len(words_norm) < 2:
        return False, ''

    w1, w2 = words_norm[0], words_norm[1]

    # Получаем фонетику из error (уже вычислена в smart_compare)
    phon_sim = error.get('phonetic_similarity', error.get('similarity', 0))
    # Нормализуем: может быть 0-100 или 0-1
    if phon_sim > 1:
        phon_sim = phon_sim / 100

    # 1. same_phonetic_diff_lemma
    should_filter, reason = check_same_phonetic_diff_lemma(w1, w2)
    if should_filter:
        return True, reason

    # 2. high_phon_sem_diff_lemma
    should_filter, reason = check_high_phon_sem_diff_lemma(w1, w2, phon_sim)
    if should_filter:
        return True, reason

    # 3. perfect_phon_diff_lemma
    should_filter, reason = check_perfect_phon_diff_lemma(w1, w2, phon_sim)
    if should_filter:
        return True, reason

    return False, ''


# =============================================================================
# СТАТИСТИКА
# =============================================================================

def get_phonetic_semantic_stats() -> Dict[str, Any]:
    """Возвращает статистику модуля."""
    return {
        'version': VERSION,
        'has_pymorphy': HAS_PYMORPHY,
        'has_semantic': HAS_SEMANTIC_MANAGER,
        'min_word_length': MIN_WORD_LENGTH,
        'high_phon_threshold': HIGH_PHON_THRESHOLD,
        'high_sem_threshold': HIGH_SEM_THRESHOLD,
        'perfect_phon_threshold': PERFECT_PHON_THRESHOLD,
        'protected_possessive_pairs': len(PROTECTED_POSSESSIVE_PAIRS),
        'protected_roots': len(PROTECTED_DIFFERENT_ROOTS),
    }
