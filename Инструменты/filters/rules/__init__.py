"""
Модули правил фильтрации v1.2.

Разделение engine.py на логические модули:
- protection.py v1.1 — защитные слои (HARD_NEGATIVES, semantic_slip)
- phonetics.py v1.0 — фонетические пары Яндекса
- alignment.py v1.0 — артефакты выравнивания
- insertion.py v1.1 — правила для insertion ошибок
- deletion.py v1.1 — правила для deletion ошибок
- substitution.py v1.1 — правила для substitution ошибок

v1.2 (2026-01-31): Убраны fallback-блоки, пороги из config.py
v1.1 (2026-01-31): Добавлены insertion.py, deletion.py, substitution.py
v1.0 (2026-01-30): Начальная версия
"""

__version__ = '1.2.0'

from .protection import (
    check_hard_negatives,
    check_semantic_slip,
    apply_protection_layers,
    # v1.2: Пороги теперь функции из config.py
    get_semantic_slip_threshold,
    get_phonetic_slip_threshold,
)

from .phonetics import (
    check_yandex_phonetic_pair,
    check_i_ya_confusion,
    YANDEX_PHONETIC_PAIRS,
    I_YA_VERB_ENDINGS,
)

from .alignment import (
    check_alignment_artifact,
    check_alignment_artifact_length,
    check_alignment_artifact_substring,
    check_safe_ending_transition,
    check_single_consonant_artifact,
    SAFE_ENDING_TRANSITIONS,
    COMPOUND_PARTICLES,
    SINGLE_CONSONANT_ARTIFACTS,
)

# v1.1: Новые модули по типам ошибок
from .insertion import (
    check_insertion_rules,
    check_split_name_insertion,
    check_compound_particle_to,
    check_interrogative_split_to,
    check_split_suffix_insertion,
    check_split_word_fragment,
    check_yandex_split_insertions,
    check_misrecognition_artifact,
    check_unknown_word_artifact,
    check_split_compound_insertion,
    check_split_word_insertion,
    COMPOUND_PREFIXES as INSERTION_COMPOUND_PREFIXES,
    DIRECTION_WORDS,
)

from .deletion import (
    check_deletion_rules,
    check_alignment_start_artifact,
    check_character_name_unrecognized,
    check_interjection_deletion,
    check_rare_adverb_deletion,
    check_sentence_start_weak,
    check_hyphenated_part,
    check_compound_word_part,
    check_alignment_artifacts_del,
    check_short_weak_words,
    check_weak_conjunctions,
)

from .substitution import (
    check_substitution_rules,
    check_yandex_merge_artifact,
    check_yandex_truncate_artifact,
    check_yandex_expand_artifact,
    check_weak_words_identical,
    check_weak_words_same_lemma,
    check_sentence_start_conjunction,
    check_identical_normalized,
    check_homophone,
    check_compound_word,
    check_merged_word,
    check_case_form,
    check_adverb_adjective,
    check_short_full_adjective,
    check_verb_gerund_safe,
    check_yandex_typical,
    check_yandex_name,
)

__all__ = [
    # Protection
    'check_hard_negatives',
    'check_semantic_slip',
    'apply_protection_layers',
    'get_semantic_slip_threshold',  # v1.2: функции вместо констант
    'get_phonetic_slip_threshold',
    # Phonetics
    'check_yandex_phonetic_pair',
    'check_i_ya_confusion',
    'YANDEX_PHONETIC_PAIRS',
    'I_YA_VERB_ENDINGS',
    # Alignment
    'check_alignment_artifact',
    'check_alignment_artifact_length',
    'check_alignment_artifact_substring',
    'check_safe_ending_transition',
    'check_single_consonant_artifact',
    'SAFE_ENDING_TRANSITIONS',
    'COMPOUND_PARTICLES',
    'SINGLE_CONSONANT_ARTIFACTS',
    # Insertion (v1.1)
    'check_insertion_rules',
    'check_split_name_insertion',
    'check_compound_particle_to',
    'check_interrogative_split_to',
    'check_split_suffix_insertion',
    'check_split_word_fragment',
    'check_yandex_split_insertions',
    'check_misrecognition_artifact',
    'check_unknown_word_artifact',
    'check_split_compound_insertion',
    'check_split_word_insertion',
    'INSERTION_COMPOUND_PREFIXES',
    'DIRECTION_WORDS',
    # Deletion (v1.1)
    'check_deletion_rules',
    'check_alignment_start_artifact',
    'check_character_name_unrecognized',
    'check_interjection_deletion',
    'check_rare_adverb_deletion',
    'check_sentence_start_weak',
    'check_hyphenated_part',
    'check_compound_word_part',
    'check_alignment_artifacts_del',
    'check_short_weak_words',
    'check_weak_conjunctions',
    # Substitution (v1.1)
    'check_substitution_rules',
    'check_yandex_merge_artifact',
    'check_yandex_truncate_artifact',
    'check_yandex_expand_artifact',
    'check_weak_words_identical',
    'check_weak_words_same_lemma',
    'check_sentence_start_conjunction',
    'check_identical_normalized',
    'check_homophone',
    'check_compound_word',
    'check_merged_word',
    'check_case_form',
    'check_adverb_adjective',
    'check_short_full_adjective',
    'check_verb_gerund_safe',
    'check_yandex_typical',
    'check_yandex_name',
]
