"""
FrequencyManager v1.1 — Менеджер частотного словаря для скоринга ошибок.

Использует частотный словарь НКРЯ (freqrnc2011.csv) для определения
редких и книжных слов. Редкие авторские слова важнее для проверки.

Пороги частотности (ipm — instances per million):
    - freq < 10 ipm: редкое слово (+40 баллов в скоринге)
    - freq < 50 ipm: книжное слово (+20 баллов)
    - freq >= 50 ipm: обычное слово (0 баллов)

Особенности:
    - Ленивая загрузка (словарь грузится при первом обращении)
    - Кэширование результатов для быстрого доступа
    - Поддержка POS-тегов для омонимов (а_CONJ vs а_INTJ)

v1.1.0 (2026-01-31): Пороги из config.py
v1.0.0 (2026-01-30): Начальная версия
"""

import os
from functools import lru_cache
from typing import Optional

VERSION = '1.1.0'
VERSION_DATE = '2026-01-31'

# v1.1: Пороги из config.py
from .config import get_freq_rare_threshold, get_freq_bookish_threshold

# Алиасы для обратной совместимости
RARE_THRESHOLD = 10      # Используй get_freq_rare_threshold()
BOOKISH_THRESHOLD = 50   # Используй get_freq_bookish_threshold()

# Путь к словарю относительно корня проекта
DEFAULT_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data',
    'freqrnc2011.csv'
)


class FrequencyManager:
    """
    Менеджер частотного словаря.

    Использование:
        fm = FrequencyManager()
        freq = fm.get_frequency('способ')  # 45.3 ipm

        if fm.is_rare('антрацит'):
            score += 40  # редкое авторское слово
    """

    def __init__(self, dict_path: Optional[str] = None):
        self.dict_path = dict_path or DEFAULT_DICT_PATH
        self._data: dict[str, float] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка словаря."""
        if self._loaded:
            return

        if not os.path.exists(self.dict_path):
            print(f"[FrequencyManager] Словарь не найден: {self.dict_path}")
            self._loaded = True
            return

        try:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        lemma = parts[0].lower()
                        pos = parts[1].lower()
                        try:
                            freq = float(parts[2])
                        except ValueError:
                            continue

                        # Ключ с POS для омонимов
                        key_with_pos = f"{lemma}_{pos}"

                        # Сохраняем максимальную частоту для леммы
                        if lemma not in self._data or freq > self._data[lemma]:
                            self._data[lemma] = freq

                        # Сохраняем частоту с POS
                        if key_with_pos not in self._data or freq > self._data[key_with_pos]:
                            self._data[key_with_pos] = freq

            print(f"[FrequencyManager] Загружено {len(self._data)} записей")
        except Exception as e:
            print(f"[FrequencyManager] Ошибка загрузки: {e}")

        self._loaded = True

    @lru_cache(maxsize=10000)
    def get_frequency(self, word: str, pos: Optional[str] = None) -> float:
        """
        Получить частотность слова (ipm).

        Args:
            word: Слово или лемма
            pos: Часть речи (опционально, для омонимов)

        Returns:
            Частотность в ipm или 0.0 если слово не найдено
        """
        self._ensure_loaded()

        word_lower = word.lower()

        # Сначала пробуем с POS (точнее для омонимов)
        if pos:
            pos_lower = pos.lower()
            key_with_pos = f"{word_lower}_{pos_lower}"
            if key_with_pos in self._data:
                return self._data[key_with_pos]

        # Затем без POS
        return self._data.get(word_lower, 0.0)

    def is_rare(self, word: str, pos: Optional[str] = None) -> bool:
        """Редкое слово (freq < rare_threshold ipm)."""
        return self.get_frequency(word, pos) < get_freq_rare_threshold()

    def is_bookish(self, word: str, pos: Optional[str] = None) -> bool:
        """Книжное слово (rare_threshold <= freq < bookish_threshold ipm)."""
        freq = self.get_frequency(word, pos)
        return get_freq_rare_threshold() <= freq < get_freq_bookish_threshold()

    def is_common(self, word: str, pos: Optional[str] = None) -> bool:
        """Обычное слово (freq >= bookish_threshold ipm)."""
        return self.get_frequency(word, pos) >= get_freq_bookish_threshold()

    def get_category(self, word: str, pos: Optional[str] = None) -> str:
        """
        Получить категорию частотности слова.

        Returns:
            'rare' | 'bookish' | 'common' | 'unknown'
        """
        freq = self.get_frequency(word, pos)
        rare = get_freq_rare_threshold()
        bookish = get_freq_bookish_threshold()

        if freq == 0.0:
            return 'unknown'
        elif freq < rare:
            return 'rare'
        elif freq < bookish:
            return 'bookish'
        else:
            return 'common'

    def get_stats(self) -> dict:
        """Получить статистику словаря."""
        self._ensure_loaded()
        return {
            'total_entries': len(self._data),
            'path': self.dict_path,
            'loaded': self._loaded,
            'rare_threshold': get_freq_rare_threshold(),
            'bookish_threshold': get_freq_bookish_threshold(),
        }


# Глобальный экземпляр (ленивая инициализация)
_manager: Optional[FrequencyManager] = None


def get_frequency_manager() -> FrequencyManager:
    """Получить глобальный экземпляр FrequencyManager."""
    global _manager
    if _manager is None:
        _manager = FrequencyManager()
    return _manager


def get_word_frequency(word: str, pos: Optional[str] = None) -> float:
    """Удобная функция для получения частотности."""
    return get_frequency_manager().get_frequency(word, pos)


def is_rare_word(word: str, pos: Optional[str] = None) -> bool:
    """Удобная функция для проверки редкости."""
    return get_frequency_manager().is_rare(word, pos)
