"""
Пакет фильтрации ошибок транскрипции — Golden Filter v8.1

Модульная архитектура:
- engine.py v9.1 — движок фильтрации (оркестратор)
- morpho_rules.py v1.1 — консервативные морфологические правила
- comparison.py v6.1 — функции сравнения слов + phonetic_normalize
- detectors.py v3.0 — специализированные детекторы
- constants.py v4.0 — словари и константы
- base.py — ABC-интерфейс FilterRule для расширяемых правил

Модульные правила (rules/):
- rules/protection.py — HARD_NEGATIVES, семантическая защита
- rules/phonetics.py — фонетические пары (ну↔но, не↔ни, и↔я)
- rules/alignment.py — артефакты выравнивания

Защитные слои:
- semantic_manager.py v2.0 — Navec семантика (защита оговорок)
- scoring_engine.py v1.2 — адаптивные штрафы (HARD_NEGATIVES)
- character_guard.py v1.0 — защита имён персонажей

Smart Filter модули (АНАЛИТИКА — не влияют на фильтрацию):
- smart_scorer.py v3.0 — накопительный скоринг (метрики для отладки)
- frequency_manager.py v1.0 — частотный словарь НКРЯ (103K слов)
- sliding_window.py v1.0 — фонетическое сравнение без пробелов
- smart_filter.py v3.0 — интеграция Smart модулей (для rebuild_smart_data.py)
- window_verifier.py v1.1 — верификация сегментов

ВАЖНО: Smart модули предоставляют метрики (smart_score, smart_rules),
но НЕ влияют на решение should_filter(). Основная фильтрация — engine.py.

Удалено:
- smart_rules.py (v11.7.2) — функционал в morpho_rules.py + rules/
- learned_rules.py (v11.7.0) — неиспользуемый

v8.2 изменения (2026-01-30):
- Документирован статус Smart модулей (аналитика, не фильтрация)
- SKIP_SPLIT_FRAGMENT перемещён в constants.py
- deprecated_filters.py — архив отключённых фильтров

v8.1 изменения (2026-01-30):
- Удалён smart_rules.py (36KB deprecated кода)
- Добавлена документация rules/ модулей
- Обновлены версии модулей
"""

__version__ = '8.2.0'
__version_date__ = '2026-01-30'

# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================
# Только эти функции/классы рекомендованы для внешнего использования.
# Остальное — внутреннее API, может измениться без предупреждения.

__all__ = [
    # --- Основной API фильтрации ---
    'should_filter_error',   # Решение по одной ошибке
    'filter_errors',         # Фильтрация списка ошибок
    'filter_report',         # Фильтрация JSON-отчёта

    # --- Морфология и сравнение ---
    'normalize_word',        # Нормализация слова
    'get_lemma',             # Получить лемму
    'get_pos',               # Получить часть речи
    'is_lemma_match',        # Проверка совпадения лемм
    'is_homophone_match',    # Проверка омофонов
    'phonetic_normalize',    # Фонетическая нормализация

    # --- Детекторы ---
    'is_yandex_typical_error',   # Типичная ошибка Яндекса
    'is_yandex_name_error',      # Ошибка на именах
    'is_compound_word_match',    # Составные слова

    # --- Константы ---
    'HOMOPHONES',            # Словарь омофонов
    'YANDEX_TYPICAL_ERRORS', # Типичные ошибки Яндекса
    'CHARACTER_NAMES',       # Имена персонажей
    'PROTECTED_WORDS',       # Защищённые слова

    # --- Расширяемость ---
    'FilterRule',            # Базовый класс для правил
    'FilterContext',         # Контекст фильтрации

    # --- Smart Filter ---
    'SmartScorer',           # Скоринг ошибок
    'SmartFilter',           # Интеграция Smart модулей
    'FrequencyManager',      # Частотность слов

    # --- Флаги доступности ---
    'HAS_PYMORPHY',          # Доступен ли pymorphy
]

# Реэкспорт основного API
from .engine import should_filter_error, filter_errors, filter_report
from .comparison import (
    normalize_word, levenshtein_distance, levenshtein_ratio,
    is_homophone_match, is_grammar_ending_match, is_case_form_match,
    is_adverb_adjective_match, is_verb_gerund_safe_match,
    is_short_full_adjective_match, is_lemma_match,
    is_similar_by_levenshtein, is_yandex_typical_error,
    is_prefix_variant, is_interjection,
    get_word_info, get_lemma, get_pos, get_number, get_gender,
    parse_word_cached,
    phonetic_normalize,  # v8.0: консолидировано из smart_rules.py
    HAS_PYMORPHY, HAS_RAPIDFUZZ,
)
from .detectors import (
    is_yandex_name_error, is_merged_word_error, is_compound_word_match,
    is_split_name_insertion, is_compound_prefix_insertion,
    is_split_compound_insertion, is_context_artifact,
    detect_alignment_chains, detect_linked_prefix_errors,
    load_character_names_dictionary, load_base_character_names,
    FULL_CHARACTER_NAMES, CHARACTER_NAMES_BASE,
)
from .constants import (
    HOMOPHONES, GRAMMAR_ENDINGS, WEAK_WORDS, PROTECTED_WORDS,
    INTERJECTIONS, YANDEX_TYPICAL_ERRORS, YANDEX_NAME_ERRORS,
    YANDEX_PREFIX_ERRORS, CHARACTER_NAMES, SAFE_TRANSPOSITIONS,
)
from .base import (
    FilterRule, FilterContext,
    register_rule, unregister_rule, get_registered_rules,
    apply_registered_rules,
)
# v11.7.2: smart_rules.py УДАЛЁН — функционал в morpho_rules.py + rules/
from .character_guard import (
    CharacterGuard, get_character_guard,
    is_character_name, is_anchor_candidate, get_word_penalty,
    COMMON_TERMS,
)
from .scoring_engine import (
    ScoringEngine, get_scoring_engine, PenaltyResult,
    calculate_penalty, should_filter_by_score, is_hard_negative,
    HARD_NEGATIVES,
)
from .window_verifier import (
    WindowVerifier, get_window_verifier, VerificationStatus, VerificationResult,
    verify_segment, is_technical_noise, is_word_transposition,
)

# v7.0: Smart Filter модули
from .smart_scorer import (
    SmartScorer, ScoreResult, get_smart_scorer,
    WEIGHTS, DEFAULT_THRESHOLD,
)
from .frequency_manager import (
    FrequencyManager, get_frequency_manager, get_word_frequency, is_rare_word,
    RARE_THRESHOLD, BOOKISH_THRESHOLD,
)
from .sliding_window import (
    SlidingWindow, SlidingResult, get_sliding_window,
    is_alignment_artifact, check_phonetic_match,
)
from .smart_filter import (
    SmartFilter, SmartFilterResult, get_smart_filter, evaluate_error_smart,
)
