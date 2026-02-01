"""
ScoringEngine v1.3 — Система адаптивных штрафов для фильтрации.

Логика:
1. Имена персонажей получают высокий штраф (+100) — не фильтруем
2. Разная морфология (лемма, POS) — штраф, но меньше
3. Hard Negatives — известные пары путаницы, штраф +80
4. Адаптивный порог на основе confidence от Яндекса

v1.3 (2026-01-31): Унификация импортов — используем .comparison вместо ..morphology
v1.2 (2026-01-29): Очищено HARD_NEGATIVES — только имена персонажей (3 пары)
    - Убраны костыли (сотни/сотня, получится/получилось и т.д.)
    - Грамматические различия должен ловить morpho_rules
v1.1 (2026-01-29): [ОТКАЧЕНО] Расширено HARD_NEGATIVES — было костылём
v1.0 (2026-01-29): Начальная версия
"""

VERSION = '1.3.0'
VERSION_DATE = '2026-01-31'

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# v1.3: Прямой импорт CharacterGuard (без fallback)
from .character_guard import get_character_guard, is_character_name, get_word_penalty
HAS_CHARACTER_GUARD = True

# v1.3: Импорт из comparison.py — единый источник морфологии для filters/
from .comparison import get_lemma, get_pos, HAS_PYMORPHY


# =============================================================================
# HARD NEGATIVES — ТОЛЬКО имена персонажей, которые Яндекс путает
# =============================================================================

# ВАЖНО: Здесь НЕ должно быть обычных слов!
# Все грамматические различия (сотни/сотня, получится/получилось)
# должны определяться через morpho_rules автоматически.
#
# HARD_NEGATIVES — только для ИМЁН ПЕРСОНАЖЕЙ вашей книги,
# которые Яндекс распознаёт как обычные слова.
#
# v1.3 (2026-01-30): Убраны рагедон/рагидон — это вариации написания, не реальные ошибки
HARD_NEGATIVES: Dict[str, str] = {
    # Имена персонажей, которые Яндекс путает с обычными словами
    'джейра': 'вчера',      # имя Джейра → "вчера"
    # УБРАНО: рагедон/рагидон — это ошибки Яндекса, не реальные ошибки чтеца

    # Добавляйте сюда ТОЛЬКО имена из вашей книги,
    # которые Яндекс систематически путает с обычными словами
}

# Обратный словарь для быстрого поиска
HARD_NEGATIVES_REVERSE: Dict[str, str] = {v: k for k, v in HARD_NEGATIVES.items()}


# =============================================================================
# SCORING ENGINE
# =============================================================================

@dataclass
class PenaltyResult:
    """Результат расчёта штрафа."""
    total_penalty: int
    reasons: List[str]
    should_not_filter: bool  # True = не фильтровать (реальная ошибка)


class ScoringEngine:
    """
    Движок адаптивного скоринга для фильтрации ошибок.

    Принцип: чем выше штраф, тем больше вероятность реальной ошибки.
    Высокий штраф = НЕ фильтровать (показать человеку).
    """

    # Пороги штрафов
    PENALTY_CHARACTER_NAME = 100  # Имя персонажа — не фильтруем
    PENALTY_DIFFERENT_LEMMA = 40  # Разные леммы — скорее всего разные слова
    PENALTY_DIFFERENT_POS = 50    # Разная часть речи
    PENALTY_HARD_NEGATIVE = 80    # Известная пара путаницы
    PENALTY_SHORT_WORD = 30       # Короткие слова чаще ошибки

    # Порог для принятия решения
    FILTER_THRESHOLD = 50  # Если penalty >= 50, не фильтруем

    # Адаптивные пороги схожести (текст vs текст)
    THRESHOLD_HIGH_CONFIDENCE = 0.85    # confidence >= 0.9
    THRESHOLD_MEDIUM_CONFIDENCE = 0.75  # confidence 0.7-0.9
    THRESHOLD_LOW_CONFIDENCE = 0.65     # confidence < 0.7

    def __init__(self):
        self._cache: Dict[Tuple[str, str], PenaltyResult] = {}

    def calculate_penalty(self, word_author: str, word_audio: str) -> PenaltyResult:
        """
        Рассчитывает штраф за несовпадение слов.

        Args:
            word_author: слово из оригинала
            word_audio: слово из транскрипции

        Returns:
            PenaltyResult с общим штрафом и причинами
        """
        cache_key = (word_author.lower(), word_audio.lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        penalty = 0
        reasons = []

        # 1. Проверка имён персонажей
        if HAS_CHARACTER_GUARD:
            author_penalty = get_word_penalty(word_author)
            audio_penalty = get_word_penalty(word_audio)

            if author_penalty > 0 or audio_penalty > 0:
                penalty += max(author_penalty, audio_penalty)
                if author_penalty > 0:
                    reasons.append(f"имя_в_оригинале:{word_author}")
                if audio_penalty > 0:
                    reasons.append(f"имя_в_транскрипции:{word_audio}")

        # 2. Проверка Hard Negatives
        author_lower = word_author.lower()
        audio_lower = word_audio.lower()

        if author_lower in HARD_NEGATIVES and HARD_NEGATIVES[author_lower] == audio_lower:
            penalty += self.PENALTY_HARD_NEGATIVE
            reasons.append(f"hard_negative:{author_lower}↔{audio_lower}")
        elif audio_lower in HARD_NEGATIVES_REVERSE and HARD_NEGATIVES_REVERSE[audio_lower] == author_lower:
            penalty += self.PENALTY_HARD_NEGATIVE
            reasons.append(f"hard_negative:{author_lower}↔{audio_lower}")

        # 3. Проверка морфологии (если не имя и не hard negative)
        if penalty < self.PENALTY_CHARACTER_NAME and HAS_PYMORPHY:
            lemma_author = get_lemma(word_author)
            lemma_audio = get_lemma(word_audio)

            if lemma_author != lemma_audio:
                penalty += self.PENALTY_DIFFERENT_LEMMA
                reasons.append(f"разные_леммы:{lemma_author}≠{lemma_audio}")

            pos_author = get_pos(word_author)
            pos_audio = get_pos(word_audio)

            if pos_author and pos_audio and pos_author != pos_audio:
                penalty += self.PENALTY_DIFFERENT_POS
                reasons.append(f"разные_POS:{pos_author}≠{pos_audio}")

        # 4. Штраф за короткие слова (они чаще ошибки)
        if len(word_author) <= 2 or len(word_audio) <= 2:
            penalty += self.PENALTY_SHORT_WORD
            reasons.append("короткое_слово")

        result = PenaltyResult(
            total_penalty=penalty,
            reasons=reasons,
            should_not_filter=(penalty >= self.FILTER_THRESHOLD)
        )

        self._cache[cache_key] = result
        return result

    def get_adaptive_threshold(self, confidence: float) -> float:
        """
        Возвращает адаптивный порог схожести на основе confidence от Яндекса.

        Высокий confidence = можем быть строже (выше порог).
        Низкий confidence = нужно быть мягче (ниже порог).
        """
        if confidence >= 0.9:
            return self.THRESHOLD_HIGH_CONFIDENCE
        elif confidence >= 0.7:
            return self.THRESHOLD_MEDIUM_CONFIDENCE
        else:
            return self.THRESHOLD_LOW_CONFIDENCE

    def should_filter(
        self,
        word_author: str,
        word_audio: str,
        text_similarity: float,
        phonetic_similarity: float,
        confidence: float = 0.8
    ) -> Tuple[bool, str]:
        """
        Решает, следует ли фильтровать ошибку.

        Args:
            word_author: слово из оригинала
            word_audio: слово из транскрипции
            text_similarity: схожесть текста (0-1)
            phonetic_similarity: фонетическая схожесть (0-1)
            confidence: уверенность Яндекса (0-1)

        Returns:
            (should_filter: bool, reason: str)
        """
        # 1. Рассчитываем штраф
        penalty_result = self.calculate_penalty(word_author, word_audio)

        # 2. Если высокий штраф — не фильтруем
        if penalty_result.should_not_filter:
            return (False, f"штраф={penalty_result.total_penalty}: {', '.join(penalty_result.reasons)}")

        # 3. Проверяем схожесть с адаптивным порогом
        threshold = self.get_adaptive_threshold(confidence)
        combined_similarity = max(text_similarity, phonetic_similarity)

        if combined_similarity >= threshold:
            return (True, f"схожесть={combined_similarity:.2f}≥{threshold:.2f}, confidence={confidence:.2f}")
        else:
            return (False, f"схожесть={combined_similarity:.2f}<{threshold:.2f}, confidence={confidence:.2f}")

    def get_stats(self) -> dict:
        """Возвращает статистику кэша."""
        return {
            'cache_size': len(self._cache),
            'penalties_over_threshold': sum(
                1 for r in self._cache.values() if r.should_not_filter
            )
        }


# =============================================================================
# SINGLETON И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

_scoring_engine_instance = None

def get_scoring_engine() -> ScoringEngine:
    """Возвращает глобальный экземпляр ScoringEngine."""
    global _scoring_engine_instance
    if _scoring_engine_instance is None:
        _scoring_engine_instance = ScoringEngine()
    return _scoring_engine_instance


def calculate_penalty(word_author: str, word_audio: str) -> PenaltyResult:
    """Быстрая функция для расчёта штрафа."""
    return get_scoring_engine().calculate_penalty(word_author, word_audio)


def should_filter_by_score(
    word_author: str,
    word_audio: str,
    text_similarity: float,
    phonetic_similarity: float,
    confidence: float = 0.8
) -> Tuple[bool, str]:
    """Быстрая функция для проверки фильтрации."""
    return get_scoring_engine().should_filter(
        word_author, word_audio, text_similarity, phonetic_similarity, confidence
    )


def is_hard_negative(word1: str, word2: str) -> bool:
    """Проверяет, является ли пара Hard Negative."""
    w1 = word1.lower()
    w2 = word2.lower()

    if w1 in HARD_NEGATIVES and HARD_NEGATIVES[w1] == w2:
        return True
    if w2 in HARD_NEGATIVES and HARD_NEGATIVES[w2] == w1:
        return True

    return False


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_scoring_engine():
    """Тест ScoringEngine."""
    print('=' * 60)
    print('ТЕСТ: ScoringEngine')
    print('=' * 60)

    engine = ScoringEngine()

    # Тест 1: Имя персонажа
    print('\n1. Имя персонажа (Леград):')
    result = engine.calculate_penalty('Леград', 'ленинград')
    print(f'   Штраф: {result.total_penalty}, Причины: {result.reasons}')
    print(f'   Не фильтровать: {result.should_not_filter}')

    # Тест 2: Hard Negative
    print('\n2. Hard Negative (джейра/вчера):')
    result = engine.calculate_penalty('джейра', 'вчера')
    print(f'   Штраф: {result.total_penalty}, Причины: {result.reasons}')
    print(f'   Не фильтровать: {result.should_not_filter}')

    # Тест 3: Разные леммы
    print('\n3. Разные леммы (живем/живы):')
    result = engine.calculate_penalty('живем', 'живы')
    print(f'   Штраф: {result.total_penalty}, Причины: {result.reasons}')
    print(f'   Не фильтровать: {result.should_not_filter}')

    # Тест 4: Короткое слово
    print('\n4. Короткое слово (и/я):')
    result = engine.calculate_penalty('и', 'я')
    print(f'   Штраф: {result.total_penalty}, Причины: {result.reasons}')
    print(f'   Не фильтровать: {result.should_not_filter}')

    # Тест 5: Адаптивные пороги
    print('\n5. Адаптивные пороги:')
    print(f'   confidence=0.95 → threshold={engine.get_adaptive_threshold(0.95)}')
    print(f'   confidence=0.80 → threshold={engine.get_adaptive_threshold(0.80)}')
    print(f'   confidence=0.50 → threshold={engine.get_adaptive_threshold(0.50)}')

    # Тест 6: should_filter
    print('\n6. Полная проверка should_filter:')
    should, reason = engine.should_filter(
        'средоточие', 'средоточия',
        text_similarity=0.90,
        phonetic_similarity=0.95,
        confidence=0.85
    )
    print(f'   средоточие/средоточия: filter={should}, reason={reason}')

    should, reason = engine.should_filter(
        'Леград', 'ленинград',
        text_similarity=0.70,
        phonetic_similarity=0.75,
        confidence=0.90
    )
    print(f'   Леград/ленинград: filter={should}, reason={reason}')

    print()
    print('=' * 60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('=' * 60)


if __name__ == '__main__':
    # v1.3: Запуск через python -m filters.scoring_engine из Инструменты/
    test_scoring_engine()
