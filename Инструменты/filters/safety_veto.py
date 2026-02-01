"""
SafetyVeto v2.3 — Финальный слой защиты от ложной фильтрации.

АРХИТЕКТУРА:
Все фильтры принимают решение "фильтровать/не фильтровать".
SafetyVeto — последний слой, который может НАЛОЖИТЬ ВЕТО на фильтрацию.

ПРИНЦИП:
- Фильтр решил фильтровать (True, reason)
- SafetyVeto проверяет: "А точно ли это безопасно?"
- Если есть сомнения → ВЕТО (False, 'VETO_xxx(was:reason)')

ПРОВЕРКИ ВЕТО:
1. hard_negative: известные пары путаницы из golden (100% защита)
2. semantic_slip: высокая семантика + разные леммы = оговорка чтеца
3. merged_diff_lemmas: merged ошибка с разными леммами = реальная ошибка
   - ИСКЛЮЧЕНИЕ: если original — имя персонажа (v2.3)
4. misrecognized_real_word: Яндекс исказил слово (эли→или)

v2.3 (2026-02-01): Исключение для имён персонажей в merged_diff_lemmas
v2.2 (2026-02-01): Исключение для вариантных форм сравнительной степени (-ей/-ее)
v2.1 (2026-01-31): Централизованные пороги из config.py
v2.0 (2026-01-31): Добавлен HARD_NEGATIVES (перенесён с уровня -1)
v1.0 (2026-01-31): Начальная версия — вынесено из engine.py
"""

VERSION = '2.3.0'  # v2.3: Исключение для имён персонажей
VERSION_DATE = '2026-02-01'

from typing import Dict, Any, Tuple, List, Optional

# Импорт морфологии
try:
    from .comparison import (
        get_lemma, normalize_word,
        HAS_PYMORPHY,
    )
except ImportError:
    HAS_PYMORPHY = False
    get_lemma = None
    normalize_word = lambda x: x.lower().strip() if x else ''

# Импорт семантики
try:
    from .semantic_manager import get_similarity
    HAS_SEMANTIC_MANAGER = True
except ImportError:
    HAS_SEMANTIC_MANAGER = False
    get_similarity = None

# Импорт HARD_NEGATIVES (v2.0)
try:
    from .scoring_engine import is_hard_negative, HARD_NEGATIVES
    HAS_HARD_NEGATIVES = True
except ImportError:
    HAS_HARD_NEGATIVES = False
    is_hard_negative = lambda w1, w2: False
    HARD_NEGATIVES = set()

# Импорт имён персонажей (v2.3)
try:
    from .detectors import CHARACTER_NAMES_BASE, FULL_CHARACTER_NAMES
    HAS_CHARACTER_NAMES = True
except ImportError:
    HAS_CHARACTER_NAMES = False
    CHARACTER_NAMES_BASE = set()
    FULL_CHARACTER_NAMES = set()


# =============================================================================
# ПОРОГИ (v2.1: централизованы в config.py)
# =============================================================================

# Импорт централизованных порогов
try:
    from .config import (
        get_veto_semantic_slip,
        get_veto_phonetic_exception,
        get_veto_semantic_exception,
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Семантический порог для оговорок
# Анализ БД: semantic >= 0.4 + diff_lemma = 77 реальных ошибок защищено
SEMANTIC_SLIP_THRESHOLD = get_veto_semantic_slip() if HAS_CONFIG else 0.4

# Фонетический порог для исключений из semantic_slip
# Если phon >= 0.8 и sem >= 0.5 — это артефакт ASR, не оговорка
PHONETIC_EXCEPTION_THRESHOLD = get_veto_phonetic_exception() if HAS_CONFIG else 0.8
SEMANTIC_EXCEPTION_THRESHOLD = get_veto_semantic_exception() if HAS_CONFIG else 0.5


# =============================================================================
# СЛОВАРЬ ИСКАЖЕНИЙ ЯНДЕКСА
# =============================================================================

# Частотные слова, которые Яндекс может исказить
# Формат: искажение → (оригинал, минимальная схожесть)
MISRECOGNITION_COMMON_WORDS = {
    'эли': ('или', 0.65),   # или → эли (потеря первой буквы)
    'ли': ('или', 0.65),    # или → ли
    'ило': ('или', 0.65),   # или → ило
    'али': ('или', 0.65),   # или → али
}


# =============================================================================
# ФУНКЦИИ ПРОВЕРКИ
# =============================================================================

def _check_hard_negative(
    error: Dict[str, Any],
    words_norm: List[str],
) -> Tuple[bool, str]:
    """
    Проверяет, является ли пара известной путаницей (HARD_NEGATIVE).

    HARD_NEGATIVES — пары слов из golden, которые НИКОГДА нельзя фильтровать.
    Это 100% реальные ошибки чтеца.

    Примеры:
    - ('мечтательны', 'мечтатели') — чтец ошибся
    - ('живем', 'живы') — чтец ошибся

    Args:
        error: Словарь с ошибкой
        words_norm: Нормализованные слова [wrong, correct]

    Returns:
        (True, reason) если пара в HARD_NEGATIVES → ВЕТО на фильтрацию
        (False, '') если не в списке
    """
    if not HAS_HARD_NEGATIVES:
        return False, ''

    if error.get('type') != 'substitution':
        return False, ''

    if len(words_norm) < 2:
        return False, ''

    w1, w2 = words_norm[0], words_norm[1]

    if is_hard_negative(w1, w2):
        return True, 'hard_negative'

    return False, ''


def _is_variant_comparative_form(w1: str, w2: str) -> bool:
    """
    Проверяет, являются ли слова вариантными формами сравнительной степени.

    Примеры: слабей/слабее, сильней/сильнее, быстрей/быстрее
    Это грамматически эквивалентные формы, НЕ ошибки чтеца.

    Pymorphy может давать разные леммы (слабеть vs слабый), поэтому
    нужна отдельная проверка.

    Args:
        w1, w2: Нормализованные слова

    Returns:
        True если это вариантные формы
    """
    if not w1 or not w2:
        return False

    # Слова должны быть одинаковой длины
    if len(w1) != len(w2):
        return False

    # Проверяем окончания -ей/-ее
    if (w1.endswith('ей') and w2.endswith('ее')) or (w1.endswith('ее') and w2.endswith('ей')):
        # Проверяем что основы совпадают
        if w1[:-2] == w2[:-2]:
            return True

    return False


def _check_semantic_slip(
    error: Dict[str, Any],
    words_norm: List[str],
) -> Tuple[bool, str]:
    """
    Проверяет, является ли ошибка оговоркой чтеца (semantic slip).

    Оговорка = чтец случайно сказал похожее по смыслу слово.
    Высокая семантика + разные леммы = реальная ошибка, не FP.

    Args:
        error: Словарь с ошибкой
        words_norm: Нормализованные слова [wrong, correct]

    Returns:
        (True, reason) если это оговорка → ВЕТО на фильтрацию
        (False, '') если не оговорка
    """
    if not HAS_SEMANTIC_MANAGER or not HAS_PYMORPHY:
        return False, ''

    if error.get('type') != 'substitution':
        return False, ''

    if len(words_norm) < 2:
        return False, ''

    w1, w2 = words_norm[0], words_norm[1]

    # Получаем леммы
    lemma1 = get_lemma(w1)
    lemma2 = get_lemma(w2)

    # Только для РАЗНЫХ лемм
    if not lemma1 or not lemma2 or lemma1 == lemma2:
        return False, ''

    # Проверяем семантику
    semantic_sim = get_similarity(w1, w2)

    if semantic_sim < SEMANTIC_SLIP_THRESHOLD:
        return False, ''

    # ИСКЛЮЧЕНИЯ: артефакты ASR, не оговорки
    # Получаем фонетику из error
    phon_sim = error.get('phonetic_similarity', error.get('similarity', 0))
    if phon_sim > 1:
        phon_sim = phon_sim / 100

    # Исключение 1: идеальное фонетическое совпадение (phon >= 0.99)
    # Это орфографические варианты: прочие→прочее, эта→это
    if phon_sim >= 0.99:
        return False, ''

    # Исключение 2: diff_start + high_phon + high_sem
    # Это артефакты ASR: вглубь→глубь, хотелось→захотелось
    if w1 and w2 and w1[0].lower() != w2[0].lower():
        if phon_sim >= PHONETIC_EXCEPTION_THRESHOLD and semantic_sim >= SEMANTIC_EXCEPTION_THRESHOLD:
            return False, ''

    # Исключение 3: Вариантные формы сравнительной степени (v2.2)
    # слабей/слабее, сильней/сильнее — грамматически эквивалентны
    # Pymorphy даёт разные леммы (слабеть vs слабый), но это один смысл
    if _is_variant_comparative_form(w1, w2):
        return False, ''

    # Это оговорка — ВЕТО на фильтрацию
    return True, f'semantic_slip({semantic_sim:.2f})'


def _is_character_name(word: str) -> bool:
    """
    Проверяет, является ли слово именем персонажа.

    v2.3: Используется для исключения имён из merged_diff_lemmas.
    Яндекс не знает имена персонажей и искажает их (рутгош→тот, рутгош→род).
    Это артефакты распознавания, не ошибки чтеца.

    Args:
        word: Слово для проверки

    Returns:
        True если слово — имя персонажа
    """
    if not HAS_CHARACTER_NAMES:
        return False

    w = word.lower().strip() if word else ''
    if not w:
        return False

    # Проверяем базовый список
    if w in CHARACTER_NAMES_BASE:
        return True

    # Проверяем полный словарь
    if FULL_CHARACTER_NAMES and w in FULL_CHARACTER_NAMES:
        return True

    return False


def _check_merged_diff_lemmas(
    error: Dict[str, Any],
    words_norm: List[str],
) -> Tuple[bool, str]:
    """
    Проверяет, является ли merged ошибка реальной (разные леммы).

    Merged ошибки создаются merge_adjacent_ins_del() из соседних insertion+deletion.
    Если леммы разные — это реальная ошибка чтеца, не артефакт.

    ИСКЛЮЧЕНИЕ (v2.3): Если original — имя персонажа, НЕ накладывать VETO.
    Яндекс не знает имена и искажает их: рутгош→тот, рутгош→род.
    Это артефакты распознавания, а не ошибки чтеца.

    Args:
        error: Словарь с ошибкой
        words_norm: Нормализованные слова [wrong, correct]

    Returns:
        (True, reason) если merged с разными леммами → ВЕТО на фильтрацию
        (False, '') если не merged или одинаковые леммы
    """
    if not error.get('merged_from_ins_del', False):
        return False, ''

    if not HAS_PYMORPHY or len(words_norm) < 2:
        return False, ''

    # v2.3: Исключение для имён персонажей
    # words_norm[0] = transcript (wrong), words_norm[1] = original (correct)
    original = words_norm[1] if len(words_norm) > 1 else ''
    if _is_character_name(original):
        # Имя персонажа — это артефакт распознавания, не ошибка чтеца
        return False, ''

    lemma1 = get_lemma(words_norm[0])
    lemma2 = get_lemma(words_norm[1])

    # Разные леммы = реальная ошибка чтеца
    if lemma1 and lemma2 and lemma1 != lemma2:
        return True, f'merged_diff_lemmas({lemma1}≠{lemma2})'

    return False, ''


def _check_misrecognized_real_word(
    error: Dict[str, Any],
    words_norm: List[str],
) -> Tuple[bool, str]:
    """
    Проверяет, является ли transcript искажённым распознаванием реального слова.

    Проблема: Яндекс иногда искажает распознанное слово.
    - чтец сказал "или", Яндекс распознал "эли"
    - original = "и", transcript = "эли"
    - Фильтры могут ошибочно считать это FP
    - Но на самом деле чтец СКАЗАЛ "или" вместо "и" — реальная ошибка!

    Args:
        error: Словарь с ошибкой
        words_norm: Нормализованные слова [wrong, correct]

    Returns:
        (True, reason) если это искажённое слово → ВЕТО на фильтрацию
        (False, '') если нет
    """
    if error.get('type') != 'substitution':
        return False, ''

    if len(words_norm) < 2:
        return False, ''

    transcript = words_norm[0].lower().strip()
    original = words_norm[1].lower().strip()

    # Проверяем известные искажения
    if transcript in MISRECOGNITION_COMMON_WORDS:
        real_word, min_sim = MISRECOGNITION_COMMON_WORDS[transcript]
        # Если реальное слово отличается от оригинала — это реальная ошибка
        if real_word != original:
            return True, f'misrecognized_{real_word}'

    return False, ''


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ВЕТО
# =============================================================================

def apply_safety_veto(
    error: Dict[str, Any],
    filter_decision: bool,
    filter_reason: str,
    words_norm: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Применяет финальное вето к решению фильтра.

    ЛОГИКА:
    1. Если фильтр решил НЕ фильтровать → ничего не меняем
    2. Если фильтр решил фильтровать → проверяем безопасность
    3. Если проверка выявила риск → ВЕТО (не фильтровать)

    Args:
        error: Словарь с ошибкой
        filter_decision: Решение фильтра (True = фильтровать)
        filter_reason: Причина решения фильтра
        words_norm: Нормализованные слова (опционально, вычислим если нет)

    Returns:
        (should_filter, reason) — финальное решение
    """
    # Если фильтр решил НЕ фильтровать — ничего не меняем
    if not filter_decision:
        return filter_decision, filter_reason

    # Подготавливаем words_norm если не переданы
    if words_norm is None:
        error_type = error.get('type', '')
        if error_type == 'substitution':
            word1 = error.get('wrong', '') or error.get('transcript', '')
            word2 = error.get('correct', '') or error.get('original', '')
            words_norm = [normalize_word(word1), normalize_word(word2)]
        elif error_type == 'insertion':
            word = error.get('wrong', '') or error.get('transcript', '') or error.get('word', '')
            words_norm = [normalize_word(word)]
        elif error_type == 'deletion':
            word = error.get('correct', '') or error.get('original', '') or error.get('word', '')
            words_norm = [normalize_word(word)]
        else:
            words_norm = []

    # ВЕТО 1: HARD_NEGATIVES (v2.0) — абсолютная защита
    is_hard, hard_reason = _check_hard_negative(error, words_norm)
    if is_hard:
        return False, f'VETO_{hard_reason}(was:{filter_reason})'

    # ВЕТО 2: semantic_slip
    is_slip, slip_reason = _check_semantic_slip(error, words_norm)
    if is_slip:
        return False, f'VETO_{slip_reason}(was:{filter_reason})'

    # ВЕТО 3: merged_diff_lemmas
    is_merged, merged_reason = _check_merged_diff_lemmas(error, words_norm)
    if is_merged:
        return False, f'VETO_{merged_reason}(was:{filter_reason})'

    # ВЕТО 4: misrecognized_real_word
    is_misrec, misrec_reason = _check_misrecognized_real_word(error, words_norm)
    if is_misrec:
        return False, f'VETO_{misrec_reason}(was:{filter_reason})'

    # Все проверки пройдены — фильтрация разрешена
    return filter_decision, filter_reason


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_veto_stats() -> Dict[str, Any]:
    """Возвращает статистику модуля."""
    return {
        'version': VERSION,
        'has_config': HAS_CONFIG,
        'has_hard_negatives': HAS_HARD_NEGATIVES,
        'hard_negatives_count': len(HARD_NEGATIVES),
        'has_semantic': HAS_SEMANTIC_MANAGER,
        'has_pymorphy': HAS_PYMORPHY,
        'semantic_slip_threshold': SEMANTIC_SLIP_THRESHOLD,
        'phonetic_exception_threshold': PHONETIC_EXCEPTION_THRESHOLD,
        'semantic_exception_threshold': SEMANTIC_EXCEPTION_THRESHOLD,
        'misrecognition_words': len(MISRECOGNITION_COMMON_WORDS),
    }
