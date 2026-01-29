"""
SmartFilter v3.0 — Интеграция всех модулей Smart фильтрации.

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v3.0:
    Одинаковая лемма НЕ означает "ложная ошибка"!
    Если есть грамматические различия (число, падеж, вид, POS) — это РЕАЛЬНАЯ ошибка.

Объединяет:
    - SmartScorer v3.0: Исправленный накопительный скоринг
    - FrequencyManager: Частотность слов
    - SemanticManager: Семантическая близость
    - SlidingWindow: Фонетическое сравнение
    - Морфология: pymorphy3 для лемм и POS (fallback: pymorphy2)

АРХИТЕКТУРА v3.0:
    1. Базовый скор = 0
    2. Морфология:
       - Разные леммы: +70
       - Одинаковая лемма + грамм. различия: +50 (РЕАЛЬНАЯ ошибка!)
       - Полностью одинаковая форма: -40 (FP)
    3. Частотность: редкие (+40), частые (-15)
    4. Длина слова: короткие (-10/-20)
    5. Sliding Window: артефакты (-100)
    6. Семантика: оговорки (+30)

Версия: 3.0.0
Дата: 2026-01-30
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .smart_scorer import (
    SmartScorer, ScoreResult, get_smart_scorer,
    WEIGHTS, DEFAULT_THRESHOLD,
    FREQ_RARE, FREQ_BOOKISH, FREQ_COMMON,
)
from .frequency_manager import (
    FrequencyManager, get_frequency_manager,
)
from .sliding_window import (
    SlidingWindow, get_sliding_window,
    is_alignment_artifact,
)

# Ленивый импорт тяжёлых модулей
_semantic_manager = None
_morph = None

VERSION = '3.0.0'
VERSION_DATE = '2026-01-30'


def _get_morph():
    """Ленивая загрузка pymorphy3 (fallback: pymorphy2)."""
    global _morph
    if _morph is None:
        try:
            import pymorphy3
            _morph = pymorphy3.MorphAnalyzer()
        except ImportError:
            try:
                import pymorphy2
                _morph = pymorphy2.MorphAnalyzer()
            except ImportError:
                pass
    return _morph


def _get_semantic_manager():
    """Ленивая загрузка SemanticManager."""
    global _semantic_manager
    if _semantic_manager is None:
        from .semantic_manager import SemanticManager
        _semantic_manager = SemanticManager()
    return _semantic_manager


@dataclass
class SmartFilterResult:
    """Результат умной фильтрации."""
    should_show: bool       # Показывать ошибку?
    score: int              # Итоговый скор
    threshold: int          # Порог
    applied_rules: List[str]  # Применённые правила
    is_artifact: bool       # Артефакт выравнивания?
    frequency_category: str  # rare/bookish/common/unknown
    semantic_similarity: Optional[float]  # Семантическое сходство


class SmartFilter:
    """
    Умный фильтр ошибок v11.

    Использование:
        sf = SmartFilter()
        result = sf.evaluate_error(error_dict, context_words_original, context_words_transcript)
        if result.should_show:
            # показать ошибку пользователю
        else:
            # отфильтровать как FP
    """

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        use_semantics: bool = True,
        use_frequency: bool = True,
        use_sliding_window: bool = True,
    ):
        self.threshold = threshold
        self.use_semantics = use_semantics
        self.use_frequency = use_frequency
        self.use_sliding_window = use_sliding_window

        self.scorer = get_smart_scorer(threshold)
        self.freq_manager = get_frequency_manager() if use_frequency else None
        self.sliding = get_sliding_window() if use_sliding_window else None

    def evaluate_error(
        self,
        error: Dict[str, Any],
        original_context: Optional[List[str]] = None,
        transcript_context: Optional[List[str]] = None,
    ) -> SmartFilterResult:
        """
        Оценить ошибку через накопительный скоринг v2.0.

        Args:
            error: Словарь ошибки из compared.json
            original_context: Слова оригинала вокруг ошибки
            transcript_context: Слова транскрипта вокруг ошибки

        Returns:
            SmartFilterResult с результатом оценки
        """
        error_type = error.get('type', 'substitution')

        # Извлекаем слова
        if error_type == 'substitution':
            original = error.get('correct', '') or error.get('original', '')
            transcript = error.get('wrong', '') or error.get('transcript', '')
        elif error_type == 'deletion':
            original = error.get('correct', '') or error.get('original', '') or error.get('word', '')
            transcript = ''
        elif error_type == 'insertion':
            original = ''
            transcript = error.get('wrong', '') or error.get('transcript', '') or error.get('word', '')
        else:
            original = error.get('original', '')
            transcript = error.get('transcript', '')

        # Создаём результат скоринга (базовый score = 0)
        result = self.scorer.create_result(error_type, original, transcript)

        # 1. Базовый скоринг по типу (небольшой бонус для deletion/insertion)
        self.scorer.apply_base_score(result)

        # 2. МОРФОЛОГИЯ — КЛЮЧЕВОЙ ПРИЗНАК v3.0
        same_lemma = False
        same_pos = True  # По умолчанию считаем одинаковыми
        has_grammar_diff = False  # v3.0: Грамматические различия (число, падеж, вид)

        if error_type == 'substitution' and original and transcript:
            morph = _get_morph()
            if morph:
                try:
                    # Получаем леммы и POS
                    p1 = morph.parse(original.lower())[0]
                    p2 = morph.parse(transcript.lower())[0]

                    lemma1 = p1.normal_form
                    lemma2 = p2.normal_form
                    pos1 = p1.tag.POS
                    pos2 = p2.tag.POS

                    # Сохраняем для отладки
                    result.original_lemma = lemma1
                    result.transcript_lemma = lemma2
                    result.original_pos = str(pos1) if pos1 else None
                    result.transcript_pos = str(pos2) if pos2 else None

                    same_lemma = (lemma1 == lemma2)
                    same_pos = (pos1 == pos2)

                    # v3.0: Проверяем грамматические различия при одинаковой лемме
                    if same_lemma:
                        # Число (sing/plur)
                        num1 = p1.tag.number
                        num2 = p2.tag.number
                        if num1 and num2 and num1 != num2:
                            has_grammar_diff = True

                        # Падеж (nomn/gent/datv/accs/ablt/loct)
                        case1 = p1.tag.case
                        case2 = p2.tag.case
                        if case1 and case2 and case1 != case2:
                            has_grammar_diff = True

                        # Вид глагола (perf/impf)
                        asp1 = p1.tag.aspect
                        asp2 = p2.tag.aspect
                        if asp1 and asp2 and asp1 != asp2:
                            has_grammar_diff = True

                        # Время глагола (past/pres/futr)
                        tense1 = p1.tag.tense
                        tense2 = p2.tag.tense
                        if tense1 and tense2 and tense1 != tense2:
                            has_grammar_diff = True

                        # Приставка "по-" у сравнительной степени
                        # "побольше" vs "больше" — разные формы!
                        w1_lower = original.lower()
                        w2_lower = transcript.lower()
                        if ('Cmp2' in str(p1.tag)) != ('Cmp2' in str(p2.tag)):
                            # Cmp2 = усилительная форма (побольше, получше)
                            has_grammar_diff = True
                        elif w1_lower.startswith('по') and w2_lower == w1_lower[2:]:
                            # "побольше" → "больше"
                            has_grammar_diff = True
                        elif w2_lower.startswith('по') and w1_lower == w2_lower[2:]:
                            # "больше" → "побольше"
                            has_grammar_diff = True

                except Exception:
                    pass

            # Применяем морфологический скоринг v3.0
            self.scorer.apply_morphology(result, same_lemma, same_pos, has_grammar_diff)

            # v3.0: УДАЛЁН штраф за фонетическое сходство
            # "давайте→дайте", "мигнул→моргнул" — это РЕАЛЬНЫЕ ошибки

        # 3. ДЛИНА СЛОВА — короткие часто артефакты (только substitution)
        word_to_check = transcript if transcript else original
        if word_to_check:
            self.scorer.apply_word_length(result, word_to_check)

        # 4. Sliding Window — проверка артефакта выравнивания
        is_artifact = False
        if self.use_sliding_window and self.sliding:
            orig_words = original_context or []
            trans_words = transcript_context or []

            # Добавляем текущие слова
            if original:
                orig_words = list(orig_words) + [original]
            if transcript:
                trans_words = list(trans_words) + [transcript]

            if orig_words and trans_words:
                sliding_result = self.sliding.check_artifact(orig_words, trans_words)
                if sliding_result.is_artifact:
                    is_artifact = True
                    self.scorer.apply_sliding_window(result, is_match=True)

        # 5. Частотность слова (использует лемму для поиска)
        freq_category = 'unknown'
        if self.use_frequency and self.freq_manager:
            freq_word = original if original else transcript
            if freq_word:
                # Сначала пробуем словоформу
                freq = self.freq_manager.get_frequency(freq_word)

                # Если не нашли — пробуем лемму
                if freq <= 0:
                    morph = _get_morph()
                    if morph:
                        try:
                            lemma = morph.parse(freq_word.lower())[0].normal_form
                            freq = self.freq_manager.get_frequency(lemma)
                        except Exception:
                            pass

                self.scorer.apply_frequency(result, freq)

                # Определяем категорию для отчёта
                if freq <= 0:
                    freq_category = 'unknown'
                elif freq < FREQ_RARE:
                    freq_category = 'rare'
                elif freq < FREQ_BOOKISH:
                    freq_category = 'bookish'
                elif freq > FREQ_COMMON:
                    freq_category = 'very_common'
                else:
                    freq_category = 'common'

        # 6. Семантическое сходство (только для substitution с разными леммами)
        semantic_sim = None
        if self.use_semantics and error_type == 'substitution' and original and transcript:
            # Семантика полезна только если леммы разные
            if not same_lemma:
                try:
                    sm = _get_semantic_manager()
                    semantic_sim = sm.similarity(original, transcript)
                    result.semantic_similarity = semantic_sim

                    if semantic_sim > 0.6:
                        self.scorer.apply_semantics(result, semantic_sim)
                except Exception:
                    pass  # Модель не загружена

        # Формируем результат
        return SmartFilterResult(
            should_show=result.is_visible(self.threshold),
            score=result.score,
            threshold=self.threshold,
            applied_rules=result.applied_rules,
            is_artifact=is_artifact,
            frequency_category=freq_category,
            semantic_similarity=semantic_sim,
        )

    def evaluate_batch(
        self,
        errors: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], SmartFilterResult]]:
        """
        Оценить пакет ошибок.

        Returns:
            Список пар (error, SmartFilterResult)
        """
        results = []
        for error in errors:
            # Извлекаем контекст из ошибки
            original_ctx = error.get('context', '').split()[:5]
            transcript_ctx = error.get('transcript_context', '').split()[:5]

            result = self.evaluate_error(error, original_ctx, transcript_ctx)
            results.append((error, result))

        return results

    def filter_errors(
        self,
        errors: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        Отфильтровать ошибки с накопительным скорингом.

        Returns:
            (visible_errors, filtered_errors, stats)
        """
        visible = []
        filtered = []
        stats = {}

        for error, result in self.evaluate_batch(errors):
            # Собираем статистику по правилам
            for rule in result.applied_rules:
                rule_name = rule.split(' (')[0]  # Убираем "(+XX)"
                stats[rule_name] = stats.get(rule_name, 0) + 1

            if result.should_show:
                visible.append(error)
            else:
                error_with_reason = {**error, 'smart_filter_reason': result.applied_rules}
                filtered.append(error_with_reason)

        stats['visible'] = len(visible)
        stats['filtered'] = len(filtered)

        return visible, filtered, stats


# Глобальный экземпляр
_smart_filter: Optional[SmartFilter] = None


def get_smart_filter(threshold: int = DEFAULT_THRESHOLD) -> SmartFilter:
    """Получить глобальный экземпляр SmartFilter."""
    global _smart_filter
    if _smart_filter is None or _smart_filter.threshold != threshold:
        _smart_filter = SmartFilter(threshold=threshold)
    return _smart_filter


def evaluate_error_smart(
    error: Dict[str, Any],
    threshold: int = DEFAULT_THRESHOLD,
) -> SmartFilterResult:
    """Удобная функция для оценки одной ошибки."""
    return get_smart_filter(threshold).evaluate_error(error)


# =============================================================================
# ТЕСТИРОВАНИЕ
# =============================================================================

def test_smart_filter():
    """Тест SmartFilter."""
    print('=' * 60)
    print('ТЕСТ: SmartFilter v1.0')
    print('=' * 60)

    sf = SmartFilter(use_semantics=False)  # Без семантики для быстрого теста

    # Тест 1: Substitution
    print('\n1. Substitution ("способа" → "выхода"):')
    error1 = {
        'type': 'substitution',
        'correct': 'способа',
        'wrong': 'выхода',
        'context': 'искать способа выбраться',
    }
    result1 = sf.evaluate_error(error1)
    print(f'   should_show: {result1.should_show}')
    print(f'   score: {result1.score} (threshold={result1.threshold})')
    print(f'   rules: {result1.applied_rules}')
    print(f'   freq: {result1.frequency_category}')

    # Тест 2: Deletion редкого слова
    print('\n2. Deletion редкого слова ("антрацит"):')
    error2 = {
        'type': 'deletion',
        'correct': 'антрацит',
        'context': 'цвета антрацит',
    }
    result2 = sf.evaluate_error(error2)
    print(f'   should_show: {result2.should_show}')
    print(f'   score: {result2.score}')
    print(f'   rules: {result2.applied_rules}')
    print(f'   freq: {result2.frequency_category}')

    # Тест 3: Insertion обычного слова
    print('\n3. Insertion обычного слова ("и"):')
    error3 = {
        'type': 'insertion',
        'wrong': 'и',
        'context': 'быстро и короткими',
    }
    result3 = sf.evaluate_error(error3)
    print(f'   should_show: {result3.should_show}')
    print(f'   score: {result3.score}')
    print(f'   rules: {result3.applied_rules}')
    print(f'   freq: {result3.frequency_category}')

    # Тест 4: Артефакт выравнивания
    print('\n4. Артефакт ("и сам он" → "исамон"):')
    error4 = {
        'type': 'substitution',
        'correct': 'сам',
        'wrong': 'исамон',
        'context': 'и сам он знал',
    }
    result4 = sf.evaluate_error(
        error4,
        original_context=['и', 'сам', 'он'],
        transcript_context=['исамон'],
    )
    print(f'   should_show: {result4.should_show}')
    print(f'   is_artifact: {result4.is_artifact}')
    print(f'   score: {result4.score}')
    print(f'   rules: {result4.applied_rules}')

    # Тест 5: Batch filter
    print('\n5. Batch filter (3 ошибки):')
    errors = [error1, error2, error3]
    visible, filtered, stats = sf.filter_errors(errors)
    print(f'   visible: {len(visible)}')
    print(f'   filtered: {len(filtered)}')
    print(f'   stats: {stats}')

    print()
    print('=' * 60)
    print('ТЕСТ ЗАВЕРШЁН')
    print('=' * 60)


if __name__ == '__main__':
    test_smart_filter()
