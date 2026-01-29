"""
Пакет фильтрации ошибок транскрипции — Golden Filter v7.0

Модульная архитектура:
- base.py — ABC-интерфейс FilterRule для расширяемых правил
- constants.py — словари и константы
- comparison.py — функции сравнения слов
- detectors.py — специализированные детекторы
- engine.py — движок фильтрации
- smart_rules.py — умные правила на основе морфологии (v6.0)
- morpho_rules.py — консервативные морфологические правила (v1.1)
- character_guard.py — защита имён персонажей (v1.0)
- scoring_engine.py — система адаптивных штрафов (v1.2)
- window_verifier.py — верификация сегментов (v1.1)

v7.0 Smart Filter модули (2026-01-30):
- smart_scorer.py v3.0 — накопительный скоринг с grammar_change
- frequency_manager.py v1.0 — частотный словарь НКРЯ (103K слов)
- sliding_window.py v1.0 — фонетическое сравнение без пробелов
- smart_filter.py v3.0 — интеграция всех Smart модулей
"""

__version__ = '7.0.0'
__version_date__ = '2026-01-30'

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
from .smart_rules import (
    SmartRules, RuleResult,
    get_smart_rules, is_smart_false_positive, get_false_positive_reason,
    phonetic_normalize,
)
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
