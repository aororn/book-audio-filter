"""
SemanticManager v2.0 — Менеджер семантических векторов для детекции оговорок.

Использует Navec (500K слов, 300d) для вычисления семантической
близости между словами. Позволяет отличить:
    - Оговорку чтеца (слова близки семантически: способ↔выход)
    - Ошибку STT (слова далеки: антрацит↔антракт)

Порог семантической близости:
    - similarity > 0.5: вероятно оговорка чтеца
    - similarity <= 0.5: скорее ошибка STT

Особенности:
    - Ленивая загрузка модели (50 МБ)
    - Кэширование результатов similarity
    - Использует navec вместо gensim (совместимость с Python 3.14)

Модель: Navec hudlit v1 (500K слов, 300d)
Версия: 2.0.0
Дата: 2026-01-30
"""

import os
from functools import lru_cache
from typing import Optional

import numpy as np

VERSION = '2.0.0'
VERSION_DATE = '2026-01-30'

# Порог семантической близости для оговорок
SYNONYM_THRESHOLD = 0.5

# Путь к модели относительно корня проекта
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'models',
    'navec_hudlit_v1_12B_500K_300d_100q.tar'
)


class SemanticManager:
    """
    Менеджер семантических векторов на основе Navec.

    Использование:
        sm = SemanticManager()
        sim = sm.similarity('способ', 'выход')  # 0.42

        if sm.is_semantic_slip('вас', 'нас'):
            score += 30  # оговорка чтеца
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._loaded = False
        self._load_error = None

    def _ensure_loaded(self) -> bool:
        """Ленивая загрузка модели."""
        if self._loaded:
            return self._model is not None

        if not os.path.exists(self.model_path):
            print(f"[SemanticManager] Модель не найдена: {self.model_path}")
            self._loaded = True
            self._load_error = "Model file not found"
            return False

        try:
            from navec import Navec
            print(f"[SemanticManager] Загрузка Navec ({os.path.getsize(self.model_path) // 1024 // 1024} МБ)...")
            self._model = Navec.load(self.model_path)
            print(f"[SemanticManager] Загружено {len(self._model.vocab.words)} слов")
        except ImportError:
            print("[SemanticManager] navec не установлен: pip install navec")
            self._load_error = "navec not installed"
        except Exception as e:
            print(f"[SemanticManager] Ошибка загрузки: {e}")
            self._load_error = str(e)

        self._loaded = True
        return self._model is not None

    def _has_word(self, word: str) -> bool:
        """Проверить, есть ли слово в модели."""
        if self._model is None:
            return False
        return word.lower() in self._model

    @lru_cache(maxsize=10000)
    def similarity(
        self,
        word1: str,
        word2: str,
        pos1: Optional[str] = None,
        pos2: Optional[str] = None,
    ) -> float:
        """
        Вычислить косинусное сходство между словами.

        Args:
            word1: Первое слово
            word2: Второе слово
            pos1: POS-тег первого слова (не используется в navec)
            pos2: POS-тег второго слова (не используется в navec)

        Returns:
            Сходство от -1.0 до 1.0, или 0.0 если слова не найдены
        """
        if not self._ensure_loaded():
            return 0.0

        w1 = word1.lower()
        w2 = word2.lower()

        if w1 not in self._model or w2 not in self._model:
            return 0.0

        try:
            v1 = self._model[w1]
            v2 = self._model[w2]
            # Косинусное сходство
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))
        except Exception:
            return 0.0

    def is_semantic_slip(
        self,
        word1: str,
        word2: str,
        pos1: Optional[str] = None,
        pos2: Optional[str] = None,
        threshold: float = SYNONYM_THRESHOLD,
    ) -> bool:
        """
        Проверить, является ли замена семантической оговоркой.

        Семантическая оговорка — когда чтец заменил слово на
        близкое по смыслу (вас → нас, теряю → теряя).

        Returns:
            True если similarity > threshold
        """
        return self.similarity(word1, word2, pos1, pos2) > threshold

    def has_word(self, word: str, pos: Optional[str] = None) -> bool:
        """Проверить, есть ли слово в модели."""
        if not self._ensure_loaded():
            return False
        return self._has_word(word)

    def get_stats(self) -> dict:
        """Получить статистику модели."""
        stats = {
            'path': self.model_path,
            'loaded': self._loaded,
            'ready': self._model is not None,
            'error': self._load_error,
            'synonym_threshold': SYNONYM_THRESHOLD,
        }
        if self._model is not None:
            stats['vocabulary_size'] = len(self._model.vocab.words)
            stats['vector_size'] = 300  # navec hudlit v1
        return stats


# Глобальный экземпляр (ленивая инициализация)
_manager: Optional[SemanticManager] = None


def get_semantic_manager() -> SemanticManager:
    """Получить глобальный экземпляр SemanticManager."""
    global _manager
    if _manager is None:
        _manager = SemanticManager()
    return _manager


def get_similarity(
    word1: str,
    word2: str,
    pos1: Optional[str] = None,
    pos2: Optional[str] = None,
) -> float:
    """Удобная функция для получения сходства."""
    return get_semantic_manager().similarity(word1, word2, pos1, pos2)


def is_semantic_slip(
    word1: str,
    word2: str,
    pos1: Optional[str] = None,
    pos2: Optional[str] = None,
) -> bool:
    """Удобная функция для проверки оговорки."""
    return get_semantic_manager().is_semantic_slip(word1, word2, pos1, pos2)
