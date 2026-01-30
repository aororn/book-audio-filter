"""
Правила артефактов выравнивания v1.0.

Артефакты — это ложные ошибки, возникающие из-за неточностей
алгоритма выравнивания текста с транскрипцией.

Типы артефактов:
- Длина: короткое слово vs длинное
- Подстрока: одно слово является частью другого
- Безопасные окончания: грамматические вариации
- Однобуквенные согласные: -с, -м, -в

v1.0 (2026-01-30): Извлечено из engine.py v8.9
"""

from typing import Tuple, Set, Optional

# Безопасные переходы окончаний (v8.8)
# Выявлены анализом БД: встречаются ТОЛЬКО в FP, НИКОГДА в Golden
SAFE_ENDING_TRANSITIONS: Set[Tuple[str, str]] = {
    # Существительные на -ие/-ия (падежи): 14 FP, 0 Golden
    ('ие', 'ия'), ('ия', 'ие'),
    # Прилагательные число/род: 11 FP, 0 Golden
    ('ые', 'ое'), ('ая', 'ые'), ('ое', 'ые'), ('ые', 'ый'), ('ый', 'ой'),
    # Глаголы 3л. ед/мн: 11 FP, 0 Golden
    ('ит', 'ят'), ('ят', 'ит'), ('ют', 'ет'), ('ет', 'ют'),
    # Существительные число: 8 FP, 0 Golden
    ('на', 'ны'), ('ну', 'ны'), ('ма', 'мы'),
    # Прилагательные падежи: 4 FP, 0 Golden
    ('ий', 'ии'), ('ии', 'ия'), ('ой', 'ны'),
    # Существительные -ье/-ья (зелье/зелья): 4 FP, 0 Golden
    ('ья', 'ье'), ('ье', 'ья'),
    # Существительные -ей/-ли (мыслей/мысли): 2 FP, 0 Golden
    ('ей', 'ли'),
    # Существительные -ью/-ти (костью/кости): 2 FP, 0 Golden
    ('ью', 'ти'),
}

# Частицы в составных словах — НЕ артефакты (v8.3)
COMPOUND_PARTICLES: Set[str] = {'то', 'нибудь', 'либо', 'кое', 'таки'}

# Однобуквенные согласные — артефакты выравнивания (v8.7)
# Исключены: я, и (есть golden), а, о, у (могут быть союзами/междометиями)
SINGLE_CONSONANT_ARTIFACTS: Set[str] = {'с', 'в', 'м', 'п', 'к', 'ф', 'х', 'э'}


def check_alignment_artifact_length(w1: str, w2: str) -> Tuple[bool, str]:
    """
    Проверяет артефакт по длине: короткое слово vs длинное.

    Паттерн: слово ≤2 символов vs слово ≥5 символов.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)

    Returns:
        (should_filter, reason)
    """
    # Исключаем частицы составных слов
    if w1 in COMPOUND_PARTICLES or w2 in COMPOUND_PARTICLES:
        return False, ''

    len1, len2 = len(w1), len(w2)

    # Короткое (≤2) vs длинное (≥5)
    if (len1 <= 2 and len2 >= 5) or (len2 <= 2 and len1 >= 5):
        return True, 'alignment_artifact_length'

    return False, ''


def check_alignment_artifact_substring(
    w1: str,
    w2: str,
    get_lemma_func: Optional[callable] = None
) -> Tuple[bool, str]:
    """
    Проверяет артефакт подстроки: одно слово является частью другого.

    v8.4: Если леммы равны — это грамматика, НЕ артефакт.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        get_lemma_func: Функция получения леммы (опционально)

    Returns:
        (should_filter, reason)
    """
    # Исключаем частицы составных слов
    if w1 in COMPOUND_PARTICLES or w2 in COMPOUND_PARTICLES:
        return False, ''

    len1, len2 = len(w1), len(w2)

    # Только для слов ≥3 символов
    if len1 >= 3 and len2 >= 3:
        if w1 in w2 or w2 in w1:
            if abs(len1 - len2) >= 2:
                # v8.4: Проверяем леммы
                if get_lemma_func:
                    lemma1 = get_lemma_func(w1)
                    lemma2 = get_lemma_func(w2)
                    if lemma1 and lemma2 and lemma1 == lemma2:
                        # Одинаковые леммы = грамматика, НЕ артефакт
                        return False, ''

                return True, 'alignment_artifact_substring'

    return False, ''


def check_safe_ending_transition(
    w1: str,
    w2: str,
    get_lemma_func: Optional[callable] = None,
    get_pos_func: Optional[callable] = None
) -> Tuple[bool, str]:
    """
    Проверяет безопасные переходы окончаний.

    Условия:
    - same_lemma = True
    - same_POS = True
    - Переход окончаний в SAFE_ENDING_TRANSITIONS

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        get_lemma_func: Функция получения леммы
        get_pos_func: Функция получения части речи

    Returns:
        (should_filter, reason)
    """
    if len(w1) < 3 or len(w2) < 3:
        return False, ''

    end1 = w1[-2:]
    end2 = w2[-2:]

    if (end1, end2) not in SAFE_ENDING_TRANSITIONS:
        return False, ''

    # Дополнительная проверка: same_lemma и same_POS
    if get_lemma_func and get_pos_func:
        lemma1 = get_lemma_func(w1)
        lemma2 = get_lemma_func(w2)
        pos1 = get_pos_func(w1)
        pos2 = get_pos_func(w2)

        if lemma1 and lemma2 and lemma1 == lemma2 and pos1 == pos2:
            return True, 'safe_ending_transition'

    return False, ''


def check_single_consonant_artifact(word: str) -> Tuple[bool, str]:
    """
    Проверяет однобуквенные согласные — артефакты выравнивания.

    Args:
        word: Слово (нормализованное)

    Returns:
        (should_filter, reason)
    """
    if len(word) == 1 and word in SINGLE_CONSONANT_ARTIFACTS:
        return True, 'single_consonant_artifact'

    return False, ''


def check_alignment_artifact(
    w1: str,
    w2: str,
    error_type: str = 'substitution',
    get_lemma_func: Optional[callable] = None,
    get_pos_func: Optional[callable] = None
) -> Tuple[bool, str]:
    """
    Проверяет все типы артефактов выравнивания.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)
        error_type: Тип ошибки
        get_lemma_func: Функция получения леммы
        get_pos_func: Функция получения части речи

    Returns:
        (should_filter, reason)
    """
    if error_type == 'substitution':
        # Артефакт по длине
        should_filter, reason = check_alignment_artifact_length(w1, w2)
        if should_filter:
            return True, reason

        # Артефакт подстроки
        should_filter, reason = check_alignment_artifact_substring(w1, w2, get_lemma_func)
        if should_filter:
            return True, reason

        # Безопасные окончания
        should_filter, reason = check_safe_ending_transition(w1, w2, get_lemma_func, get_pos_func)
        if should_filter:
            return True, reason

    elif error_type in ('deletion', 'insertion'):
        word = w1 if w1 else w2
        # Однобуквенные согласные
        should_filter, reason = check_single_consonant_artifact(word)
        if should_filter:
            return True, reason

    return False, ''
