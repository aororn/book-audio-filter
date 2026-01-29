"""
SmartScorer v3.0 — Исправленная система накопительного скоринга.

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v3.0:
    Одинаковая лемма НЕ означает "ложная ошибка"!
    Если лемма одинаковая, но есть грамматические различия — это РЕАЛЬНАЯ ошибка.

ФИЛОСОФИЯ v3.0:
    - score = 0 (базовый)
    - Накапливаем ПРИЗНАКИ РЕАЛЬНОСТИ (+)
    - Вычитаем ПРИЗНАКИ ЛОЖНОСТИ (-)
    - score >= threshold (60) = показать ошибку

ПРИЗНАКИ РЕАЛЬНОСТИ (добавляют баллы):
    +100: Имя персонажа (character_shield)
    +70:  Разные леммы (different_lemma) — разные слова! (увеличено)
    +50:  Грамматическое различие при одной лемме (grammar_change)
          - VERB↔GRND (теряю→теряя)
          - Разное число (сотни→сотня)
          - Разный падеж (ключ→ключа)
          - Разный вид (получится→получилось)
    +40:  Разная часть речи (pos_mismatch)
    +40:  Редкое слово (rare_word, freq < 10 ipm)
    +30:  Семантическая оговорка (semantic_slip, sim > 0.6)
    +20:  Книжное слово (bookish_word, freq < 50 ipm)

ПРИЗНАКИ ЛОЖНОСТИ (вычитают баллы):
    -100: Sliding Window артефакт
    -40:  Одинаковая форма слова (same_form) — только если ВСЁ одинаково!
    -15:  Очень частое слово (freq > 500 ipm)
    -10:  Короткое слово (len <= 2)

ПОРОГ: 60

Версия: 3.0.0
Дата: 2026-01-30
"""

from dataclasses import dataclass, field
from typing import Optional

VERSION = '3.0.0'
VERSION_DATE = '2026-01-30'

# =============================================================================
# ВЕСА СКОРИНГА v3.0
# =============================================================================

WEIGHTS = {
    # === ПРИЗНАКИ РЕАЛЬНОСТИ (добавляют баллы) ===

    # Критические
    'character_shield': 100,  # Имя персонажа

    # Лингвистические признаки реальной ошибки
    'different_lemma': 70,    # Разные леммы = разные слова!
    'grammar_change': 70,     # Грамматическое изменение при той же лемме (УВЕЛИЧЕНО до 70)
    'pos_mismatch': 40,       # Разная часть речи (дополнительно к grammar_change)

    # Частотность (редкие слова = авторский стиль)
    'rare_word': 40,          # freq < 10 ipm (редкие)
    'bookish_word': 20,       # freq < 50 ipm (книжные)

    # Семантика
    'semantic_slip': 30,      # Оговорка (similarity > 0.6)

    # Тип ошибки
    'deletion_base': 40,      # Пропуск слова (УВЕЛИЧЕНО с 30)
    'insertion_base': 40,     # Вставка слова (УВЕЛИЧЕНО с 30)
    'substitution': 0,        # Замена — нейтрально

    # Бонусы для deletion/insertion по частотности
    'del_ins_common': 30,     # Пропуск/вставка ЧАСТОГО слова
    'del_ins_rare': -30,      # Пропуск/вставка РЕДКОГО слова (ОСЛАБЛЕНО с -50)

    # === ПРИЗНАКИ ЛОЖНОСТИ (вычитают баллы) ===

    # Артефакты выравнивания
    'sliding_window_match': -100,  # Фонетическое совпадение без пробелов

    # Морфологические признаки FP — ТОЛЬКО когда форма ПОЛНОСТЬЮ идентична
    'same_form': -40,         # Одинаковая лемма + POS + граммемы (НОВЫЙ, заменяет same_lemma)
    'same_pos': 0,            # УБРАН штраф — одинаковая POS сама по себе ничего не значит

    # Частотность — УБРАН штраф
    # Причина: "или→и", "вас→нас", "она→они" — это РЕАЛЬНЫЕ ошибки!
    'very_common_word_sub': 0,    # УБРАН (был -15)

    # Длина слова — УБРАН штраф
    # Причина: "а→и", "мы→вы" — это РЕАЛЬНЫЕ ошибки!
    'short_word_sub': 0,          # УБРАН (был -10)
    'very_short_word_sub': 0,     # УБРАН (был -20)

    # УДАЛЕНЫ:
    # 'same_lemma': -60 — НЕПРАВИЛЬНО! Одинаковая лемма + грамматика = реальная ошибка
    # 'phonetic_similar_weak': -30 — НЕПРАВИЛЬНО! "давайте→дайте" это реальная ошибка
}

# Порог видимости
DEFAULT_THRESHOLD = 60

# Частотные пороги
FREQ_RARE = 10        # ipm — редкое слово
FREQ_BOOKISH = 50     # ipm — книжное слово
FREQ_COMMON = 500     # ipm — очень частое слово


@dataclass
class ScoreResult:
    """Результат скоринга для одной ошибки."""

    error_type: str  # 'substitution', 'deletion', 'insertion'
    original: str    # Слово из оригинала
    transcript: str  # Слово из транскрипта

    score: int = 0
    applied_rules: list = field(default_factory=list)
    is_technical_artifact: bool = False

    # Дополнительные данные для отладки
    original_lemma: Optional[str] = None
    transcript_lemma: Optional[str] = None
    original_pos: Optional[str] = None
    transcript_pos: Optional[str] = None
    frequency: Optional[float] = None
    semantic_similarity: Optional[float] = None

    def add_score(self, points: int, rule_name: str) -> None:
        """Добавить баллы от правила."""
        self.score += points
        sign = '+' if points >= 0 else ''
        self.applied_rules.append(f"{rule_name} ({sign}{points})")

    def is_visible(self, threshold: int = DEFAULT_THRESHOLD) -> bool:
        """Должна ли ошибка показываться в отчёте."""
        if self.is_technical_artifact:
            return False
        return self.score >= threshold

    def get_explanation(self) -> str:
        """Получить объяснение скоринга."""
        rules = ', '.join(self.applied_rules) if self.applied_rules else 'нет правил'
        return f"Score={self.score}: {rules}"


class SmartScorer:
    """
    Накопительный скорер для ошибок v3.0.

    КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Одинаковая лемма НЕ означает FP!
    - "сотни→сотня" (разное число) = РЕАЛЬНАЯ ошибка
    - "теряю→теряя" (VERB→GRND) = РЕАЛЬНАЯ ошибка
    - "ключ→ключа" (разный падеж) = РЕАЛЬНАЯ ошибка
    """

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.weights = WEIGHTS.copy()

    def create_result(
        self,
        error_type: str,
        original: str,
        transcript: str,
    ) -> ScoreResult:
        """Создать новый результат скоринга."""
        return ScoreResult(
            error_type=error_type,
            original=original,
            transcript=transcript,
        )

    def apply_base_score(self, result: ScoreResult) -> None:
        """Применить базовый скоринг по типу ошибки."""
        if result.error_type == 'deletion':
            result.add_score(self.weights['deletion_base'], 'deletion_base')
        elif result.error_type == 'insertion':
            result.add_score(self.weights['insertion_base'], 'insertion_base')
        # substitution = 0, нейтрально

    def apply_character_shield(self, result: ScoreResult, is_character_name: bool) -> None:
        """Применить защиту имён персонажей."""
        if is_character_name:
            result.add_score(self.weights['character_shield'], 'character_shield')

    def apply_morphology(
        self,
        result: ScoreResult,
        same_lemma: bool,
        same_pos: bool,
        has_grammar_diff: bool = False,
    ) -> None:
        """
        Применить морфологический скоринг v3.0.

        КЛЮЧЕВОЕ ИЗМЕНЕНИЕ:
        - Разные леммы = разные слова = +70 (реальная ошибка)
        - Одинаковая лемма + грамматические различия = +50 (РЕАЛЬНАЯ ошибка!)
        - Одинаковая лемма БЕЗ грамматических различий = -40 (FP)
        - Разная POS = +40

        Args:
            same_lemma: Одинаковая ли лемма
            same_pos: Одинаковая ли часть речи
            has_grammar_diff: Есть ли грамматические различия (число, падеж, вид, время)
        """
        if not same_lemma:
            # Разные леммы = разные слова = реальная ошибка
            result.add_score(self.weights['different_lemma'], 'different_lemma')
        else:
            # Одинаковая лемма — проверяем грамматику
            if has_grammar_diff or not same_pos:
                # Грамматические различия = РЕАЛЬНАЯ ошибка!
                # Примеры: сотни→сотня, теряю→теряя, ключ→ключа
                result.add_score(self.weights['grammar_change'], 'grammar_change')
            else:
                # Полностью одинаковая форма = вероятно FP
                result.add_score(self.weights['same_form'], 'same_form')

        # Части речи — бонус только за РАЗНЫЕ
        if not same_pos:
            result.add_score(self.weights['pos_mismatch'], 'pos_mismatch')
        # Убран штраф за same_pos — это не признак FP

    def apply_frequency(self, result: ScoreResult, freq_ipm: float) -> None:
        """Применить скоринг по частотности слова."""
        result.frequency = freq_ipm

        if result.error_type == 'substitution':
            # Для substitution: редкое слово = важная ошибка
            if freq_ipm <= 0:
                result.add_score(self.weights['rare_word'], 'unknown_word')
            elif freq_ipm < FREQ_RARE:
                result.add_score(self.weights['rare_word'], 'rare_word')
            elif freq_ipm < FREQ_BOOKISH:
                result.add_score(self.weights['bookish_word'], 'bookish_word')
            elif freq_ipm > FREQ_COMMON:
                result.add_score(self.weights['very_common_word_sub'], 'very_common_word')
        else:
            # Для deletion/insertion
            if freq_ipm <= 0 or freq_ipm < FREQ_RARE:
                result.add_score(self.weights['del_ins_rare'], 'del_ins_rare_word')
            elif freq_ipm > FREQ_COMMON:
                result.add_score(self.weights['del_ins_common'], 'del_ins_common_word')

    def apply_word_length(self, result: ScoreResult, word: str) -> None:
        """Применить скоринг по длине слова."""
        if not word:
            return

        # Штраф ТОЛЬКО для substitution
        if result.error_type != 'substitution':
            return

        word_len = len(word)
        if word_len == 1:
            result.add_score(self.weights['very_short_word_sub'], 'very_short_word')
        elif word_len <= 2:
            result.add_score(self.weights['short_word_sub'], 'short_word')

    def apply_phonetic_similarity(self, result: ScoreResult, ratio: float) -> None:
        """
        v3.0: УДАЛЁН штраф за фонетическое сходство.

        Причина: "давайте→дайте", "мигнул→моргнул" — это РЕАЛЬНЫЕ ошибки,
        не нужно их штрафовать за похожее звучание.
        """
        # НЕ применяем штраф — это было ошибкой в v2.0
        pass

    def apply_semantics(self, result: ScoreResult, similarity: float) -> None:
        """Применить семантический скоринг."""
        result.semantic_similarity = similarity
        if similarity > 0.6:
            result.add_score(self.weights['semantic_slip'], 'semantic_slip')

    def apply_sliding_window(self, result: ScoreResult, is_match: bool) -> None:
        """Применить результат плавающего окна (артефакт выравнивания)."""
        if is_match:
            result.is_technical_artifact = True
            result.add_score(self.weights['sliding_window_match'], 'sliding_window_artifact')


def get_smart_scorer(threshold: int = DEFAULT_THRESHOLD) -> SmartScorer:
    """Получить экземпляр SmartScorer."""
    return SmartScorer(threshold=threshold)
