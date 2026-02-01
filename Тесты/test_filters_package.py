#!/usr/bin/env python3
"""
Тесты для пакета filters/ v5.0

Покрывает:
- filters.comparison — нормализация, Левенштейн, омофоны, грамматика, лемматизация
- filters.detectors — имена, составные слова, контекстные артефакты, цепочки
- filters.engine — should_filter_error, filter_errors, filter_report
- filters.constants — наличие обязательных словарей

Запуск:
    pytest Тесты/test_filters_package.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))

import pytest

# =============================================================================
# ИМПОРТЫ ИЗ ПАКЕТА filters
# =============================================================================

from filters import __version__
from filters.comparison import (
    normalize_word,
    levenshtein_distance, levenshtein_ratio,
    is_homophone_match, is_grammar_ending_match, is_case_form_match,
    is_adverb_adjective_match, is_verb_gerund_safe_match,
    is_short_full_adjective_match, is_lemma_match,
    is_similar_by_levenshtein, is_yandex_typical_error,
    is_prefix_variant, is_interjection,
    get_word_info, get_lemma, get_pos, get_number, get_gender,
    parse_word_cached,
    HAS_PYMORPHY, HAS_RAPIDFUZZ,
)
from filters.detectors import (
    is_yandex_name_error, is_merged_word_error, is_compound_word_match,
    is_split_name_insertion, is_compound_prefix_insertion,
    is_split_compound_insertion, is_context_artifact,
    detect_alignment_chains,
    FULL_CHARACTER_NAMES, CHARACTER_NAMES_BASE,
)
from filters.engine import (
    should_filter_error, filter_errors, filter_report,
)
from filters.constants import (
    HOMOPHONES, GRAMMAR_ENDINGS, WEAK_WORDS, PROTECTED_WORDS,
    INTERJECTIONS, YANDEX_TYPICAL_ERRORS, YANDEX_NAME_ERRORS,
    YANDEX_PREFIX_ERRORS, CHARACTER_NAMES,
)


# =============================================================================
# ТЕСТЫ ВЕРСИИ
# =============================================================================

class TestVersion:
    """Тесты версии пакета."""

    def test_version_format(self):
        assert __version__ == '5.7.0'

    def test_version_is_string(self):
        assert isinstance(__version__, str)


# =============================================================================
# ТЕСТЫ КОНСТАНТ
# =============================================================================

class TestConstants:
    """Тесты наличия обязательных словарей."""

    def test_homophones_non_empty(self):
        assert len(HOMOPHONES) > 0

    def test_grammar_endings_non_empty(self):
        assert len(GRAMMAR_ENDINGS) > 0

    def test_weak_words_non_empty(self):
        assert len(WEAK_WORDS) > 0

    def test_interjections_non_empty(self):
        assert len(INTERJECTIONS) > 0

    def test_yandex_typical_errors_non_empty(self):
        assert len(YANDEX_TYPICAL_ERRORS) > 0

    def test_yandex_name_errors_non_empty(self):
        assert len(YANDEX_NAME_ERRORS) > 0

    def test_homophones_are_tuples(self):
        """Омофоны хранятся как множество кортежей."""
        for item in list(HOMOPHONES)[:5]:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_protected_words_are_set(self):
        assert isinstance(PROTECTED_WORDS, set)

    def test_character_names_is_set(self):
        assert isinstance(CHARACTER_NAMES, set)


# =============================================================================
# ТЕСТЫ COMPARISON — НОРМАЛИЗАЦИЯ
# =============================================================================

class TestNormalizeWord:
    """Тесты нормализации слов."""

    def test_lowercase(self):
        assert normalize_word("СЛОВО") == "слово"

    def test_strip_whitespace(self):
        assert normalize_word("  слово  ") == "слово"

    def test_yo_replacement(self):
        assert normalize_word("ёлка") == "елка"
        assert normalize_word("ЁЛКА") == "елка"

    def test_combined(self):
        assert normalize_word("  ЁЖИК  ") == "ежик"

    def test_empty(self):
        assert normalize_word("") == ""

    def test_spaces_only(self):
        assert normalize_word("   ") == ""

    def test_idempotent(self):
        word = "ЁЖИК"
        first = normalize_word(word)
        second = normalize_word(first)
        assert first == second


# =============================================================================
# ТЕСТЫ COMPARISON — ЛЕВЕНШТЕЙН
# =============================================================================

class TestLevenshtein:
    """Тесты расстояния Левенштейна."""

    def test_identical(self):
        assert levenshtein_distance("слово", "слово") == 0

    def test_one_char(self):
        assert levenshtein_distance("кот", "код") == 1

    def test_two_chars(self):
        assert levenshtein_distance("слово", "слива") == 2

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_ratio_identical(self):
        ratio = levenshtein_ratio("слово", "слово")
        assert ratio == 100

    def test_ratio_different(self):
        ratio = levenshtein_ratio("абв", "где")
        assert ratio < 50

    def test_similar_by_levenshtein(self):
        assert is_similar_by_levenshtein("слово", "слова") is True
        assert is_similar_by_levenshtein("слово", "совсем") is False

    def test_adaptive_threshold_short(self):
        """Для коротких слов порог = 1."""
        assert is_similar_by_levenshtein("кот", "код") is True
        assert is_similar_by_levenshtein("да", "до") is True


# =============================================================================
# ТЕСТЫ COMPARISON — ОМОФОНЫ
# =============================================================================

class TestHomophones:
    """Тесты омофонов."""

    def test_identical(self):
        assert is_homophone_match("слово", "слово") is True

    def test_case_insensitive(self):
        assert is_homophone_match("Слово", "СЛОВО") is True

    def test_basic_pairs(self):
        assert is_homophone_match("его", "ево") is True
        assert is_homophone_match("что", "што") is True
        assert is_homophone_match("ну", "но") is True

    def test_ya_i(self):
        # v5.6: я↔и убран из HOMOPHONES — теперь фильтруется контекстно в engine.py
        # потому что это семантически разные слова (местоимение vs союз)
        assert is_homophone_match("я", "и") is False
        assert is_homophone_match("и", "я") is False

    def test_not_homophones(self):
        assert is_homophone_match("дом", "кот") is False

    def test_symmetry(self):
        assert is_homophone_match("его", "ево") == is_homophone_match("ево", "его")

    def test_empty_strings(self):
        assert is_homophone_match("", "") is True
        assert is_homophone_match("слово", "") is False


# =============================================================================
# ТЕСТЫ COMPARISON — ГРАММАТИЧЕСКИЕ ОКОНЧАНИЯ
# =============================================================================

class TestGrammarEndings:
    """Тесты грамматических окончаний."""

    def test_adjective_gender(self):
        assert is_grammar_ending_match("красный", "красное") is True

    def test_verb_number(self):
        assert is_grammar_ending_match("заметит", "заметят") is True

    def test_case_endings(self):
        # v5.6: Падежные окончания ('ей', 'и') ЕСТЬ в GRAMMAR_ENDINGS
        # линией↔линии, печатей↔печати фильтруются как незначительные различия
        assert is_grammar_ending_match("печатей", "печати") is True
        assert is_grammar_ending_match("линией", "линии") is True

    def test_short_words(self):
        assert is_grammar_ending_match("да", "но") is False

    def test_different_words(self):
        assert is_grammar_ending_match("дом", "кот") is False


# =============================================================================
# ТЕСТЫ COMPARISON — МОРФОЛОГИЯ (требует pymorphy2)
# =============================================================================

@pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy2 not installed")
class TestMorphologyFunctions:
    """Тесты морфологических функций."""

    def test_get_lemma(self):
        lemma = get_lemma("домов")
        assert lemma == "дом"

    def test_get_pos(self):
        pos = get_pos("красный")
        assert pos in ('ADJF', 'ADJS')

    def test_get_number(self):
        num = get_number("дом")
        assert num == 'sing'

    def test_get_gender(self):
        gender = get_gender("красный")
        assert gender == 'masc'

    def test_get_word_info_tuple(self):
        info = get_word_info("домов")
        assert isinstance(info, tuple)
        assert len(info) == 5

    def test_parse_word_cached(self):
        result1 = parse_word_cached("дом")
        result2 = parse_word_cached("дом")
        assert result1 == result2  # кэшировано

    def test_case_form_match(self):
        # "печати" vs "печатей" — разное число, теперь НЕ фильтруется
        assert is_case_form_match("печати", "печатей") is False
        assert is_case_form_match("красный", "красить") is False

    def test_adverb_adjective_match(self):
        result = is_adverb_adjective_match("довольны", "довольно")
        assert isinstance(result, bool)
        assert is_adverb_adjective_match("дом", "кот") is False

    def test_short_full_adjective(self):
        result = is_short_full_adjective_match("уверены", "уверенный")
        assert isinstance(result, bool)
        assert is_short_full_adjective_match("дом", "кот") is False

    def test_verb_gerund_safe(self):
        result = is_verb_gerund_safe_match("оборачиваясь", "оборачиваясь")
        assert isinstance(result, bool)

    def test_lemma_match(self):
        # "дом" vs "дома" — разные падежи, теперь НЕ фильтруется
        assert is_lemma_match("дом", "дома") is False
        assert is_lemma_match("дом", "кот") is False

    def test_prefix_variant(self):
        result = is_prefix_variant("думать", "подумать")
        assert isinstance(result, bool)


# =============================================================================
# ТЕСТЫ COMPARISON — ОШИБКИ ЯНДЕКСА
# =============================================================================

class TestYandexErrors:
    """Тесты типичных ошибок Яндекса."""

    def test_typical_errors(self):
        assert is_yandex_typical_error("сто", "то") is True
        assert is_yandex_typical_error("изза", "из-за") is True

    def test_not_typical(self):
        assert is_yandex_typical_error("дом", "кот") is False

    def test_symmetric(self):
        """Пары проверяются в обе стороны."""
        assert is_yandex_typical_error("изза", "из-за") is True
        assert is_yandex_typical_error("из-за", "изза") is True


# =============================================================================
# ТЕСТЫ COMPARISON — МЕЖДОМЕТИЯ
# =============================================================================

class TestInterjections:
    """Тесты междометий."""

    def test_single_char(self):
        assert is_interjection("п") is True
        assert is_interjection("ф") is True

    def test_multi_char(self):
        assert is_interjection("ах") is True
        assert is_interjection("ох") is True
        assert is_interjection("хм") is True

    def test_not_interjections(self):
        assert is_interjection("а") is False
        assert is_interjection("о") is False

    def test_empty(self):
        assert is_interjection("") is False


# =============================================================================
# ТЕСТЫ DETECTORS — ИМЕНА
# =============================================================================

class TestNameDetectors:
    """Тесты детекторов имён."""

    def test_yandex_name_error_known(self):
        """Известная ошибка имени."""
        assert is_yandex_name_error("дорогалым", "дарагалом") is True

    def test_garbled_names(self):
        assert is_yandex_name_error("лжедорогала", "что-угодно") is True

    def test_different_words(self):
        """Обычные слова не являются ошибками имён."""
        result = is_yandex_name_error("стол", "стул")
        assert isinstance(result, bool)

    def test_character_names_loaded(self):
        """Словари имён загружены (могут быть пустыми если нет файла)."""
        assert isinstance(FULL_CHARACTER_NAMES, set)
        assert isinstance(CHARACTER_NAMES_BASE, set)


# =============================================================================
# ТЕСТЫ DETECTORS — СОСТАВНЫЕ СЛОВА
# =============================================================================

class TestCompoundDetectors:
    """Тесты детекторов составных слов."""

    def test_compound_word_match_empty(self):
        assert is_compound_word_match("", "") is False

    def test_compound_word_match_different(self):
        assert is_compound_word_match("дом", "кот") is False

    def test_merged_word_basic(self):
        result = is_merged_word_error("какойто", "какой то слово")
        assert isinstance(result, bool)

    def test_split_name_insertion(self):
        result = is_split_name_insertion("тест", "префикс тест")
        assert isinstance(result, bool)

    def test_compound_prefix_insertion(self):
        result = is_compound_prefix_insertion("тест", "по тест")
        assert isinstance(result, bool)

    def test_split_compound_insertion(self):
        result = is_split_compound_insertion("нибудь", "кто нибудь", "кто-нибудь")
        assert isinstance(result, bool)


# =============================================================================
# ТЕСТЫ DETECTORS — КОНТЕКСТНЫЕ АРТЕФАКТЫ
# =============================================================================

class TestContextArtifact:
    """Тесты контекстных артефактов."""

    def test_repeated_conjunction(self):
        """Многократное 'и' в контексте."""
        error = {
            'type': 'deletion',
            'correct': 'и',
            'context': 'и вот и тут и там и тогда'
        }
        result = is_context_artifact(error)
        assert result is True

    def test_normal_context(self):
        error = {
            'type': 'deletion',
            'correct': 'и',
            'context': 'он пошёл и сделал'
        }
        result = is_context_artifact(error)
        assert result is False

    def test_non_deletion(self):
        error = {
            'type': 'substitution',
            'correct': 'и',
            'context': 'и вот и тут и там'
        }
        result = is_context_artifact(error)
        assert result is False


# =============================================================================
# ТЕСТЫ DETECTORS — ЦЕПОЧКИ СМЕЩЕНИЯ
# =============================================================================

class TestAlignmentChains:
    """Тесты детектора цепочек смещения."""

    def test_basic_chain(self):
        """Классическая цепочка смещения с разными словами (dist > 3)."""
        # v5.6: Цепочки фильтруются только если:
        # - слова длинные (>2 букв)
        # - И расстояние Левенштейна > 3 или > 50% длины
        errors = [
            {'time': 15.0, 'wrong': 'алгоритм', 'correct': 'компьютер', 'type': 'substitution'},
            {'time': 15.5, 'wrong': 'компьютер', 'correct': 'программа', 'type': 'substitution'},
        ]
        chain_indices = detect_alignment_chains(errors, time_window=3.0)
        assert isinstance(chain_indices, set)
        assert len(chain_indices) > 0

    def test_no_chain_far_apart(self):
        errors = [
            {'time': 10.0, 'wrong': 'a', 'correct': 'b', 'type': 'substitution'},
            {'time': 100.0, 'wrong': 'c', 'correct': 'd', 'type': 'substitution'},
        ]
        chain_indices = detect_alignment_chains(errors, time_window=3.0)
        assert len(chain_indices) == 0

    def test_empty(self):
        assert detect_alignment_chains([]) == set()

    def test_single_error(self):
        errors = [{'time': 10.0, 'wrong': 'a', 'correct': 'b', 'type': 'substitution'}]
        assert detect_alignment_chains(errors) == set()


# =============================================================================
# ТЕСТЫ ENGINE — should_filter_error
# =============================================================================

class TestShouldFilterError:
    """Тесты главной функции фильтрации."""

    def test_homophone_substitution(self):
        error = {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'}
        should_filter, reason = should_filter_error(error)
        assert should_filter is True
        assert reason == 'homophone'

    def test_yandex_typical(self):
        error = {'type': 'substitution', 'wrong': 'сто', 'correct': 'то'}
        should_filter, reason = should_filter_error(error)
        assert should_filter is True
        assert reason == 'yandex_typical'

    def test_alignment_artifact_deletion(self):
        error = {'type': 'deletion', 'correct': 'в'}
        should_filter, reason = should_filter_error(error)
        assert should_filter is True
        assert 'alignment' in reason

    def test_real_error_passes(self):
        error = {'type': 'substitution', 'wrong': 'живем', 'correct': 'живы'}
        should_filter, reason = should_filter_error(error)
        assert should_filter is False
        assert reason == 'real_error'

    def test_interjection_deletion(self):
        error = {'type': 'deletion', 'correct': 'ах'}
        should_filter, reason = should_filter_error(error)
        assert should_filter is True

    def test_empty_error(self):
        error = {'type': 'substitution', 'wrong': '', 'correct': ''}
        should_filter, reason = should_filter_error(error)
        assert isinstance(should_filter, bool)
        assert isinstance(reason, str)

    def test_unknown_type(self):
        error = {'type': 'unknown', 'wrong': 'a', 'correct': 'b'}
        should_filter, reason = should_filter_error(error)
        assert isinstance(should_filter, bool)

    def test_deterministic(self):
        error = {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'}
        r1 = should_filter_error(error)
        r2 = should_filter_error(error)
        assert r1 == r2

    def test_golden_standard_not_filtered(self):
        """Реальные ошибки чтеца не должны фильтроваться."""
        real_errors = [
            {'type': 'substitution', 'wrong': 'живем', 'correct': 'живы'},
            {'type': 'substitution', 'wrong': 'выхода', 'correct': 'способа'},
            {'type': 'substitution', 'wrong': 'громче', 'correct': 'громко'},
            {'type': 'insertion', 'wrong': 'сказать'},
        ]
        for error in real_errors:
            should_filter, reason = should_filter_error(error)
            assert should_filter is False, (
                f"Ошибка отфильтрована: {error.get('wrong', '')}→{error.get('correct', '')} ({reason})"
            )

    def test_with_config(self):
        """Фильтрация с кастомной конфигурацией."""
        error = {'type': 'substitution', 'wrong': 'слова', 'correct': 'слово'}
        config = {'levenshtein_threshold': 1, 'use_homophones': True}
        should_filter, reason = should_filter_error(error, config=config)
        assert isinstance(should_filter, bool)


# =============================================================================
# ТЕСТЫ ENGINE — filter_errors
# =============================================================================

class TestFilterErrors:
    """Тесты пакетной фильтрации."""

    def test_basic_filtering(self):
        errors = [
            {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'},
            {'type': 'substitution', 'wrong': 'живем', 'correct': 'живы'},
        ]
        # v9.7.0: filter_errors возвращает 4 элемента
        filtered, removed, stats, protected = filter_errors(errors)
        assert len(filtered) + len(removed) == len(errors)
        assert isinstance(stats, dict)
        assert isinstance(protected, dict)
        assert len(filtered) <= len(errors)
        # v9.7.0: Проверка консистентности — сумма stats == len(removed)
        assert sum(stats.values()) == len(removed)

    def test_empty_list(self):
        filtered, removed, stats, protected = filter_errors([])
        assert filtered == []
        assert removed == []
        assert isinstance(stats, dict)
        assert isinstance(protected, dict)

    def test_all_filtered(self):
        """Все ошибки — ложные срабатывания."""
        errors = [
            {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'},
            {'type': 'substitution', 'wrong': 'его', 'correct': 'ево'},
        ]
        filtered, removed, stats, protected = filter_errors(errors)
        assert len(filtered) == 0
        assert len(removed) == 2
        # v9.7.0: sum(stats) == len(removed)
        assert sum(stats.values()) == 2


# =============================================================================
# ТЕСТЫ ENGINE — filter_report
# =============================================================================

class TestFilterReport:
    """Тесты фильтрации JSON-отчёта."""

    def test_filter_report_basic(self, tmp_path):
        """Фильтрация JSON-файла."""
        report = {
            'errors': [
                {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'},
                {'type': 'substitution', 'wrong': 'живем', 'correct': 'живы'},
            ],
            'metadata': {'chapter': 'test'}
        }

        input_path = tmp_path / 'test_compared.json'
        output_path = tmp_path / 'test_filtered.json'

        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False)

        filter_report(str(input_path), output_path=str(output_path), force=True)

        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        assert 'errors' in result
        assert 'filter_metadata' in result

    def test_filter_report_nonexistent(self):
        """Несуществующий файл."""
        with pytest.raises(Exception):
            filter_report('/nonexistent/path.json')


# =============================================================================
# ТЕСТЫ ОБРАТНОЙ СОВМЕСТИМОСТИ
# =============================================================================

class TestBackwardsCompatibility:
    """Тесты обратной совместимости — импорт из golden_filter."""

    def test_import_from_golden_filter(self):
        """Все функции доступны через golden_filter."""
        from golden_filter import (
            normalize_word, should_filter_error, filter_report,
            is_homophone_match, is_grammar_ending_match,
            HOMOPHONES, GRAMMAR_ENDINGS,
        )
        assert callable(normalize_word)
        assert callable(should_filter_error)
        assert callable(filter_report)

    def test_golden_filter_same_results(self):
        """golden_filter и filters дают одинаковые результаты."""
        from golden_filter import should_filter_error as gf_filter
        from filters.engine import should_filter_error as pkg_filter

        error = {'type': 'substitution', 'wrong': 'ну', 'correct': 'но'}
        assert gf_filter(error) == pkg_filter(error)


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
