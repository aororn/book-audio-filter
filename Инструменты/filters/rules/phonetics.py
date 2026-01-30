"""
Фонетические правила фильтрации v1.0.

Правила для фонетических пар, которые Яндекс часто путает.

v1.0 (2026-01-30): Извлечено из engine.py v8.9
"""

from typing import Tuple, Set

# Фонетические пары, которые Яндекс регулярно путает
# Формат: (транскрипт, оригинал)
YANDEX_PHONETIC_PAIRS: Set[Tuple[str, str]] = {
    # Частицы не/ни
    ('не', 'ни'), ('ни', 'не'),
    # Частицы ну/но
    ('ну', 'но'), ('но', 'ну'),
    # Гласные
    ('а', 'о'), ('о', 'а'),
    ('и', 'э'), ('э', 'и'),
    ('я', 'и'), ('и', 'я'),
    # Междометия
    ('хм', 'кхм'), ('кхм', 'хм'),
    ('ах', 'ох'), ('ох', 'ах'),
    # Местоимения
    ('он', 'она'), ('она', 'он'),
}


def check_yandex_phonetic_pair(w1: str, w2: str) -> Tuple[bool, str]:
    """
    Проверяет фонетические пары Яндекса.

    Args:
        w1: Первое слово (нормализованное)
        w2: Второе слово (нормализованное)

    Returns:
        (should_filter, reason) — если should_filter=True, фильтровать
    """
    if (w1, w2) in YANDEX_PHONETIC_PAIRS:
        return True, 'yandex_phonetic_pair'

    return False, ''


# Паттерн и/я — расширенный (v5.6)
# Яндекс ОЧЕНЬ часто путает "и" и "я" в определённых контекстах
I_YA_VERB_ENDINGS = ('л', 'ла', 'ло', 'ли', 'лся', 'лась', 'лось', 'лись')
I_YA_REFLEXIVE_ENDINGS = ('сь', 'ся', 'шься', 'шись', 'юсь', 'усь')
I_YA_FIRST_PERSON_ENDINGS = ('ю', 'у')


def check_i_ya_confusion(
    w1: str,
    w2: str,
    context: str = '',
    marker_pos: int = -1
) -> Tuple[bool, str]:
    """
    Проверяет путаницу и↔я в контексте.

    Контексты, где это ложное срабатывание:
    1. Граница предложений: "сказал. Я" → "сказал и"
    2. После глагола прошедшего времени: "кричал я" → "кричал и"
    3. После возвратного глагола: "справлюсь я" → "справлюсь и"
    4. Перед глаголом 1 лица: "я надеюсь" → "и надеюсь"

    Args:
        w1: Транскрипт (что услышал Яндекс)
        w2: Оригинал (что должно быть)
        context: Контекст ошибки
        marker_pos: Позиция маркера в контексте

    Returns:
        (should_filter, reason)
    """
    # Проверяем только пару и↔я
    if not ((w1 == 'и' and w2 == 'я') or (w1 == 'я' and w2 == 'и')):
        return False, ''

    context_lower = context.lower()

    if marker_pos > 0:
        before = context_lower[:marker_pos].rstrip()
        after_part = context_lower[marker_pos:].lstrip() if marker_pos < len(context_lower) else ''
        after_words = after_part.split()[1:] if after_part.split() else []

        # Разбиваем before с учётом знаков препинания
        before_words = before.replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').split()

        if before_words:
            last_word = before_words[-1]

            # 1. Если перед и/я стоит точка — граница предложений
            if last_word in {'.', '!', '?'}:
                return True, 'yandex_i_ya_boundary'

            # 2. Глагол прошедшего времени
            if last_word.endswith(I_YA_VERB_ENDINGS):
                return True, 'yandex_i_ya_after_verb'

            # 3. Возвратный глагол
            if last_word.endswith(I_YA_REFLEXIVE_ENDINGS):
                return True, 'yandex_i_ya_after_verb'

            # 4. Глагол 1 лица на -у/-ю
            if last_word.endswith(I_YA_FIRST_PERSON_ENDINGS) and len(last_word) >= 3:
                return True, 'yandex_i_ya_after_verb'

        # 5. Перед глаголом 1 лица
        if after_words:
            next_word = after_words[0].rstrip('.,!?')
            if next_word.endswith(('ю', 'у', 'юсь', 'усь')) and len(next_word) >= 3:
                return True, 'yandex_i_ya_before_verb'

    return False, ''
