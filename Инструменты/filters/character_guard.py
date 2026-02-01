"""
CharacterGuard v1.1 — Центральный модуль защиты имён персонажей.

Используется:
- AlignmentManager — для исключения имён из якорей
- ScoringEngine — для выставления штрафов за ошибки в именах
- smart_compare — для приоритизации защиты имён

v1.1 (2026-01-31): Унификация импортов (убран fallback)
v1.0 (2026-01-29): Начальная версия
"""

VERSION = '1.1.0'
VERSION_DATE = '2026-01-31'

from typing import Set, Optional
from functools import lru_cache

# v1.1: Прямые импорты (без fallback)
from .detectors import (
    FULL_CHARACTER_NAMES,
    CHARACTER_NAMES_BASE,
    load_character_names_dictionary,
    load_base_character_names,
)
from .comparison import normalize_word


# =============================================================================
# ЧАСТЫЕ ТЕРМИНЫ (не использовать как якоря)
# =============================================================================

# Слова, которые встречаются часто в тексте и не подходят для якорения
COMMON_TERMS: Set[str] = {
    # Звания и должности
    'магистр', 'магистра', 'магистру', 'магистром', 'магистре',
    'комтур', 'комтура', 'комтуру', 'комтуром', 'комтуре',
    'предводитель', 'предводителя', 'предводителю', 'предводителем',
    'старейшина', 'старейшины', 'старейшине', 'старейшиной',

    # Организации
    'орден', 'ордена', 'ордену', 'орденом', 'ордене',
    'клан', 'клана', 'клану', 'кланом', 'клане',
    'семья', 'семьи', 'семье', 'семьей', 'семью',
    'альянс', 'альянса', 'альянсу', 'альянсом', 'альянсе',
    'империя', 'империи', 'империю', 'империей',

    # Магия и способности
    'указ', 'указа', 'указу', 'указом', 'указе', 'указы', 'указов',
    'печать', 'печати', 'печатью', 'печатей',
    'эссенция', 'эссенции', 'эссенцию', 'эссенцией',
    'сила', 'силы', 'силу', 'силой', 'силе', 'сил',
    'дух', 'духа', 'духу', 'духом', 'духе', 'духи', 'духов',

    # Существа и понятия
    'голем', 'голема', 'голему', 'големом', 'големе', 'големы', 'големов',
    'небесный', 'небесного', 'небесному', 'небесным', 'небесная', 'небесные',
    'древний', 'древнего', 'древнему', 'древним', 'древняя', 'древние', 'древних',

    # Частые глаголы и слова
    'сказал', 'сказала', 'ответил', 'ответила', 'спросил', 'спросила',
    'произнес', 'произнесла', 'добавил', 'добавила',
}


class CharacterGuard:
    """
    Центральный модуль защиты имён персонажей.

    Отвечает за:
    1. Определение является ли слово именем персонажа
    2. Определение можно ли использовать слово как якорь
    3. Расчёт штрафа за ошибки в именах
    """

    def __init__(self):
        """Инициализация с загрузкой словарей."""
        self._full_names: Set[str] = FULL_CHARACTER_NAMES
        self._base_names: Set[str] = CHARACTER_NAMES_BASE
        self._common_terms: Set[str] = COMMON_TERMS

        # Кэш для быстрой проверки
        self._cache_is_character: dict = {}
        self._cache_is_anchor: dict = {}

    @property
    def character_count(self) -> int:
        """Количество известных имён (все формы)."""
        return len(self._full_names)

    @property
    def base_names_count(self) -> int:
        """Количество базовых имён."""
        return len(self._base_names)

    def is_character(self, word: str) -> bool:
        """
        Проверяет, является ли слово именем персонажа.

        Args:
            word: Слово для проверки

        Returns:
            True если слово — имя персонажа (в любой падежной форме)
        """
        word_norm = normalize_word(word)

        if word_norm in self._cache_is_character:
            return self._cache_is_character[word_norm]

        result = word_norm in self._full_names or word_norm in self._base_names
        self._cache_is_character[word_norm] = result
        return result

    def is_base_name(self, word: str) -> bool:
        """
        Проверяет, является ли слово базовой формой имени.

        Базовые формы — это именительный падеж (Леград, Мириот, Лейла).
        """
        word_norm = normalize_word(word)
        return word_norm in self._base_names

    def is_common_term(self, word: str) -> bool:
        """
        Проверяет, является ли слово частым термином.

        Частые термины не подходят для якорения, т.к. встречаются много раз.
        """
        word_norm = normalize_word(word)
        return word_norm in self._common_terms

    def is_anchor_candidate(self, word: str, min_length: int = 6) -> bool:
        """
        Проверяет, можно ли использовать слово как якорь для выравнивания.

        Слово подходит для якоря если:
        1. Длина >= min_length символов
        2. НЕ является именем персонажа
        3. НЕ является частым термином

        Args:
            word: Слово для проверки
            min_length: Минимальная длина слова

        Returns:
            True если слово можно использовать как якорь
        """
        word_norm = normalize_word(word)

        cache_key = (word_norm, min_length)
        if cache_key in self._cache_is_anchor:
            return self._cache_is_anchor[cache_key]

        # Проверка длины
        if len(word_norm) < min_length:
            self._cache_is_anchor[cache_key] = False
            return False

        # Имена персонажей — плохие якоря (повторяются часто)
        if self.is_character(word_norm):
            self._cache_is_anchor[cache_key] = False
            return False

        # Частые термины — плохие якоря
        if self.is_common_term(word_norm):
            self._cache_is_anchor[cache_key] = False
            return False

        self._cache_is_anchor[cache_key] = True
        return True

    def get_penalty(self, word: str) -> int:
        """
        Возвращает штраф за ошибку в слове.

        Ошибки в именах персонажей караются максимально жёстко (+100 баллов),
        т.к. это критически важная информация для читателя.

        Args:
            word: Слово для проверки

        Returns:
            Штрафные баллы (0 для обычных слов, 100 для имён)
        """
        if self.is_character(word):
            return 100
        return 0

    def get_protection_level(self, word: str) -> str:
        """
        Возвращает уровень защиты слова.

        Returns:
            'high' — имя персонажа, максимальная защита
            'medium' — частый термин, средняя защита
            'low' — обычное слово
        """
        if self.is_character(word):
            return 'high'
        if self.is_common_term(word):
            return 'medium'
        return 'low'

    def add_common_term(self, term: str):
        """Добавляет термин в список частых."""
        self._common_terms.add(normalize_word(term))
        # Сбрасываем кэш якорей
        self._cache_is_anchor.clear()

    def clear_cache(self):
        """Очищает кэши."""
        self._cache_is_character.clear()
        self._cache_is_anchor.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_character_guard_instance: Optional[CharacterGuard] = None


def get_character_guard() -> CharacterGuard:
    """Возвращает глобальный экземпляр CharacterGuard."""
    global _character_guard_instance
    if _character_guard_instance is None:
        _character_guard_instance = CharacterGuard()
    return _character_guard_instance


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def is_character_name(word: str) -> bool:
    """Быстрая проверка является ли слово именем персонажа."""
    return get_character_guard().is_character(word)


def is_anchor_candidate(word: str, min_length: int = 6) -> bool:
    """Быстрая проверка можно ли использовать слово как якорь."""
    return get_character_guard().is_anchor_candidate(word, min_length)


def get_word_penalty(word: str) -> int:
    """Быстрое получение штрафа за слово."""
    return get_character_guard().get_penalty(word)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_character_guard():
    """Тест CharacterGuard на известных именах."""
    guard = get_character_guard()

    print('='*60)
    print('ТЕСТ: CharacterGuard')
    print('='*60)
    print(f'Загружено имён (все формы): {guard.character_count}')
    print(f'Загружено базовых имён: {guard.base_names_count}')
    print()

    # Тест имён персонажей
    test_names = ['Леград', 'леград', 'ЛЕГРАД', 'Мириот', 'Лейла', 'Рагедон', 'рагидон']
    print('Проверка имён персонажей:')
    for name in test_names:
        is_char = guard.is_character(name)
        penalty = guard.get_penalty(name)
        print(f'  {name}: is_character={is_char}, penalty={penalty}')
    print()

    # Тест частых терминов
    test_terms = ['магистр', 'комтур', 'орден', 'клан', 'указ']
    print('Проверка частых терминов:')
    for term in test_terms:
        is_common = guard.is_common_term(term)
        is_anchor = guard.is_anchor_candidate(term)
        print(f'  {term}: is_common_term={is_common}, is_anchor_candidate={is_anchor}')
    print()

    # Тест кандидатов на якоря
    test_anchors = ['средоточие', 'Миражный', 'Павильон', 'смертельно', 'выравнивание']
    print('Проверка кандидатов на якоря:')
    for word in test_anchors:
        is_anchor = guard.is_anchor_candidate(word)
        protection = guard.get_protection_level(word)
        print(f'  {word}: is_anchor_candidate={is_anchor}, protection={protection}')
    print()

    print('='*60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('='*60)


if __name__ == '__main__':
    test_character_guard()
