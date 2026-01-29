"""
Пакет фильтрации ошибок транскрипции — Golden Filter v6.2

Модульная архитектура:
- base.py — ABC-интерфейс FilterRule для расширяемых правил
- constants.py — словари и константы
- comparison.py — функции сравнения слов
- detectors.py — специализированные детекторы
- engine.py — движок фильтрации
- smart_rules.py — умные правила на основе морфологии (v6.0)
- morpho_rules.py — консервативные морфологические правила (v1.1)
- character_guard.py — защита имён персонажей (v1.0) [NEW v9.0]

v6.3 изменения (2026-01-29):
- Добавлен scoring_engine.py — система адаптивных штрафов
- Добавлен window_verifier.py — верификация сегментов

v6.2 изменения (2026-01-29):
- Добавлен character_guard.py — центральный модуль защиты имён
- CharacterGuard определяет: имена персонажей, кандидаты на якоря, штрафы

v6.1 изменения:
- Добавлен morpho_rules.py с консервативной фильтрацией
- Исправлен _is_proper_name() для составных слов

v6.0 изменения:
- Добавлен smart_rules.py с алгоритмическими правилами
- Расширен morphology.py (aspect, voice, tense)
- Унифицированы пути к БД через config.py
"""

__version__ = '6.3.0'
__version_date__ = '2026-01-29'

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
