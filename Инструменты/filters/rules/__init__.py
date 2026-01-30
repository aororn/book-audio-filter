"""
Модули правил фильтрации v1.0.

Разделение engine.py на логические модули:
- protection.py — защитные слои (HARD_NEGATIVES, semantic_slip)
- phonetics.py — фонетические пары Яндекса
- alignment.py — артефакты выравнивания

v1.0 (2026-01-30): Начальная версия
"""

__version__ = '1.0.0'

from .protection import (
    check_hard_negatives,
    check_semantic_slip,
    apply_protection_layers,
    SEMANTIC_SLIP_THRESHOLD,
    PHONETIC_SLIP_THRESHOLD,
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

__all__ = [
    # Protection
    'check_hard_negatives',
    'check_semantic_slip',
    'apply_protection_layers',
    'SEMANTIC_SLIP_THRESHOLD',
    'PHONETIC_SLIP_THRESHOLD',
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
]
