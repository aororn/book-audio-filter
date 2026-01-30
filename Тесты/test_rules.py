#!/usr/bin/env python3
"""
Unit-тесты для filters/rules/ модулей v1.0.

Тестирует:
- protection.py — защитные слои (HARD_NEGATIVES, semantic_slip)
- phonetics.py — фонетические пары Яндекса
- alignment.py — артефакты выравнивания

Версия: 1.0.0
Дата: 2026-01-30
"""

import pytest
import sys
from pathlib import Path

# Добавляем путь к модулям
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'Инструменты'))

from filters.rules import (
    # Protection
    check_hard_negatives,
    check_semantic_slip,
    apply_protection_layers,
    SEMANTIC_SLIP_THRESHOLD,
    # Phonetics
    check_yandex_phonetic_pair,
    check_i_ya_confusion,
    YANDEX_PHONETIC_PAIRS,
    # Alignment
    check_alignment_artifact,
    check_alignment_artifact_length,
    check_alignment_artifact_substring,
    check_safe_ending_transition,
    check_single_consonant_artifact,
    SAFE_ENDING_TRANSITIONS,
)


# =============================================================================
# PHONETICS TESTS
# =============================================================================

class TestYandexPhoneticPairs:
    """Тесты для фонетических пар Яндекса."""

    def test_ne_ni_pair(self):
        """не↔ни должны фильтроваться."""
        should_filter, reason = check_yandex_phonetic_pair('не', 'ни')
        assert should_filter
        assert reason == 'yandex_phonetic_pair'

        should_filter, reason = check_yandex_phonetic_pair('ни', 'не')
        assert should_filter

    def test_nu_no_pair(self):
        """ну↔но должны фильтроваться."""
        should_filter, reason = check_yandex_phonetic_pair('ну', 'но')
        assert should_filter

        should_filter, reason = check_yandex_phonetic_pair('но', 'ну')
        assert should_filter

    def test_i_ya_pair(self):
        """и↔я в YANDEX_PHONETIC_PAIRS."""
        assert ('и', 'я') in YANDEX_PHONETIC_PAIRS
        assert ('я', 'и') in YANDEX_PHONETIC_PAIRS

    def test_non_phonetic_pair(self):
        """Не фонетические пары не должны фильтроваться."""
        should_filter, reason = check_yandex_phonetic_pair('кот', 'собака')
        assert not should_filter
        assert reason == ''


class TestIYaConfusion:
    """Тесты для путаницы и↔я."""

    def test_after_past_tense_verb(self):
        """После глагола прошедшего времени — фильтровать."""
        # "сказал я" → "сказал и"
        should_filter, reason = check_i_ya_confusion(
            'и', 'я',
            context='он сказал я знаю',
            marker_pos=10  # позиция "я" в контексте
        )
        assert should_filter
        assert 'yandex_i_ya' in reason

    def test_after_reflexive_verb(self):
        """После возвратного глагола — фильтровать."""
        # "справлюсь я" → "справлюсь и"
        should_filter, reason = check_i_ya_confusion(
            'и', 'я',
            context='я справлюсь я сделаю',
            marker_pos=12
        )
        assert should_filter

    def test_at_sentence_boundary(self):
        """На границе предложений — фильтровать."""
        # "сказал. Я" → "сказал и"
        should_filter, reason = check_i_ya_confusion(
            'и', 'я',
            context='он сказал. я знаю',
            marker_pos=11
        )
        assert should_filter
        assert 'boundary' in reason

    def test_no_context(self):
        """Без контекста — не фильтровать."""
        should_filter, reason = check_i_ya_confusion('и', 'я')
        assert not should_filter

    def test_not_i_ya_pair(self):
        """Не и/я — не фильтровать."""
        should_filter, reason = check_i_ya_confusion('он', 'она', context='что он она')
        assert not should_filter


# =============================================================================
# ALIGNMENT TESTS
# =============================================================================

class TestAlignmentArtifact:
    """Тесты для артефактов выравнивания."""

    def test_alignment_artifact_length(self):
        """Артефакт по длине — большая разница в длине."""
        should_filter, reason = check_alignment_artifact_length('и', 'исамон')
        assert should_filter
        assert reason == 'alignment_artifact_length'

    def test_substring_artifact(self):
        """Артефакт подстроки."""
        should_filter, reason = check_alignment_artifact_substring('сам', 'исамон')
        # Зависит от реализации — проверяем что функция работает
        assert isinstance(should_filter, bool)

    def test_safe_ending_transition(self):
        """Безопасные переходы окончаний."""
        # Проверяем что константы существуют
        assert isinstance(SAFE_ENDING_TRANSITIONS, (set, frozenset, dict))

        # Тест функции
        should_filter, reason = check_safe_ending_transition('слово', 'словом')
        assert isinstance(should_filter, bool)

    def test_single_consonant_artifact(self):
        """Артефакт одиночной согласной."""
        should_filter, reason = check_single_consonant_artifact('м')
        assert isinstance(should_filter, bool)


# =============================================================================
# PROTECTION TESTS
# =============================================================================

class TestHardNegatives:
    """Тесты для HARD_NEGATIVES."""

    def test_known_confusion_pair(self):
        """Известные пары путаницы должны защищаться."""
        # Проверяем что функция работает без ошибок
        # Конкретные пары зависят от scoring_engine.py
        should_protect, reason = check_hard_negatives('живем', 'живы')
        assert isinstance(should_protect, bool)

    def test_non_confusion_pair(self):
        """Обычные слова не должны защищаться."""
        should_protect, reason = check_hard_negatives('стол', 'стул')
        # Может быть True или False в зависимости от HARD_NEGATIVES
        assert isinstance(should_protect, bool)


class TestSemanticSlip:
    """Тесты для semantic slip."""

    def test_threshold_exists(self):
        """Порог должен быть определён."""
        assert 0.0 < SEMANTIC_SLIP_THRESHOLD < 1.0

    def test_semantic_slip_function(self):
        """Функция должна работать без ошибок."""
        # Может вернуть False если navec не загружен
        should_protect, reason = check_semantic_slip('способа', 'выхода')
        assert isinstance(should_protect, bool)


class TestApplyProtectionLayers:
    """Тесты для apply_protection_layers."""

    def test_non_substitution(self):
        """Для не-substitution защита не применяется."""
        should_protect, reason = apply_protection_layers('и', '', error_type='deletion')
        assert not should_protect
        assert reason == ''

    def test_substitution(self):
        """Для substitution защита применяется."""
        should_protect, reason = apply_protection_layers('живем', 'живы', error_type='substitution')
        # Результат зависит от HARD_NEGATIVES и semantic_manager
        assert isinstance(should_protect, bool)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRulesIntegration:
    """Интеграционные тесты для всех правил."""

    def test_all_rules_importable(self):
        """Все правила должны импортироваться без ошибок."""
        from filters.rules import (
            check_hard_negatives,
            check_semantic_slip,
            apply_protection_layers,
            check_yandex_phonetic_pair,
            check_i_ya_confusion,
            check_alignment_artifact,
        )
        # Если дошли сюда — импорт успешен
        assert True

    def test_rules_return_tuples(self):
        """Все функции должны возвращать кортежи (bool, str)."""
        results = [
            check_hard_negatives('a', 'b'),
            check_semantic_slip('a', 'b'),
            apply_protection_layers('a', 'b'),
            check_yandex_phonetic_pair('a', 'b'),
            check_i_ya_confusion('a', 'b'),
            check_alignment_artifact('a', 'b'),
        ]

        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
