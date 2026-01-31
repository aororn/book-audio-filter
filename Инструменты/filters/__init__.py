"""
Пакет фильтрации ошибок транскрипции — Golden Filter v8.5

Модульная архитектура:
- engine.py v9.18 — движок фильтрации (оркестратор) + SafetyVeto
- morpho_rules.py v1.3 — консервативные морфологические правила
- comparison.py v6.4 — функции сравнения слов + phonetic_normalize
- context_verifier.py v4.2 — контекстная верификация (4 уровня)
- detectors.py v3.0 — специализированные детекторы
- constants.py v4.0 — словари и константы
- base.py v1.1 — DEPRECATED, используйте rules/

Модульные правила (rules/):
- rules/protection.py — HARD_NEGATIVES, семантическая защита
- rules/phonetics.py — фонетические пары (ну↔но, не↔ни, и↔я)
- rules/alignment.py — артефакты выравнивания

Защитные слои:
- safety_veto.py v1.0 — ФИНАЛЬНОЕ вето на фильтрацию (v8.5)
- semantic_manager.py v2.0 — Navec семантика (защита оговорок)
- scoring_engine.py v1.2 — адаптивные штрафы (HARD_NEGATIVES)
- character_guard.py v1.0 — защита имён персонажей
Инфраструктура (v8.4 — рефакторинг):
- config.py v1.0 — централизованная конфигурация и пороги
- dependencies.py v1.1 — менеджер зависимостей
- extractors.py v1.0 — экстракторы слов и контекста из ошибок

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

v8.5 изменения (2026-01-31):
- engine.py v9.18 — SafetyVeto: финальный слой защиты от ложной фильтрации
- safety_veto.py v1.0 — вынесен из engine.py: semantic_slip, merged_diff_lemmas, misrecognized

v8.4 изменения (2026-01-31):
- comparison.py v6.4 — убран sys.path hack
- morpho_rules.py v1.3 — убран sys.path hack
- context_verifier.py v4.2 — унификация pymorphy через dependencies.py
- dependencies.py v1.1 — убран sys.path hack, инициализация флагов совместимости
- base.py v1.1 — помечен как DEPRECATED

v8.3 изменения (2026-01-31):
- config.py v1.0 — централизованные пороги
- dependencies.py v1.0 — менеджер зависимостей
- extractors.py v1.0 — экстракторы данных

v8.2 изменения (2026-01-30):
- Документирован статус Smart модулей (аналитика, не фильтрация)
- SKIP_SPLIT_FRAGMENT перемещён в constants.py
- deprecated_filters.py — архив отключённых фильтров
"""

__version__ = '8.5.0'
__version_date__ = '2026-01-31'

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

    # --- Инфраструктура (v8.3) ---
    'FilterConfig',          # Конфигурация фильтрации
    'FilterThresholds',      # Пороги фильтрации
    'Dependencies',          # Менеджер зависимостей
    'get_dependencies',      # Получить глобальные зависимости
    'extract_words',         # Извлечь слова из ошибки
    'extract_context',       # Извлечь контекст из ошибки
    'ExtractedWords',        # Результат извлечения слов
    'ExtractedContext',      # Результат извлечения контекста
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
# v8.5: SafetyVeto — финальный слой защиты
from .safety_veto import (
    apply_safety_veto, get_veto_stats,
    SEMANTIC_SLIP_THRESHOLD, MISRECOGNITION_COMMON_WORDS,
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

# v8.3: Инфраструктура рефакторинга
from .config import (
    FilterConfig, FilterThresholds, FilterFlags, VersionRequirements,
    get_default_config, set_default_config,
)
from .dependencies import (
    Dependencies, DependencyStatus, get_dependencies, reload_dependencies,
)
from .extractors import (
    ExtractedWords, ExtractedContext,
    extract_words, extract_context, extract_all,
    get_error_type, get_time, is_merged_error,
    get_context_words, find_word_in_context,
    get_before_marker, get_after_marker, ends_with_sentence,
)
