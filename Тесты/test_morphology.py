#!/usr/bin/env python3
"""
Тесты для morphology.py v5.0

Покрывает:
- normalize_word — нормализация
- get_word_info / get_lemma / get_pos / get_number / get_gender / get_case
- is_same_lemma / is_same_pos
- get_all_forms
- MorphCache — персистентный кэш
- get_cache_stats / clear_cache

Запуск:
    pytest Тесты/test_morphology.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))

import pytest
from morphology import (
    normalize_word, get_word_info, get_lemma, get_pos,
    get_number, get_gender, get_case,
    is_same_lemma, is_same_pos, get_all_forms,
    parse_word_cached, get_cache_stats, clear_cache,
    MorphCache, get_disk_cache,
    HAS_PYMORPHY, VERSION,
)


# =============================================================================
# ТЕСТЫ ВЕРСИИ
# =============================================================================

class TestVersion:
    def test_version(self):
        assert VERSION == '5.0.0'


# =============================================================================
# ТЕСТЫ НОРМАЛИЗАЦИИ
# =============================================================================

class TestNormalize:
    def test_lowercase(self):
        assert normalize_word("СЛОВО") == "слово"

    def test_strip(self):
        assert normalize_word("  слово  ") == "слово"

    def test_yo(self):
        assert normalize_word("ёлка") == "елка"

    def test_empty(self):
        assert normalize_word("") == ""

    def test_idempotent(self):
        r = normalize_word("ЁЖИК")
        assert normalize_word(r) == r


# =============================================================================
# ТЕСТЫ МОРФОЛОГИЧЕСКОГО АНАЛИЗА
# =============================================================================

@pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy2 not installed")
class TestMorphAnalysis:

    def test_get_word_info_returns_tuple(self):
        info = get_word_info("дом")
        assert isinstance(info, tuple)
        assert len(info) == 5

    def test_get_lemma_noun(self):
        assert get_lemma("домов") == "дом"
        assert get_lemma("котов") == "кот"

    def test_get_lemma_verb(self):
        lemma = get_lemma("бежал")
        assert lemma in ("бежать", "бечь")

    def test_get_pos_noun(self):
        assert get_pos("дом") == 'NOUN'

    def test_get_pos_verb(self):
        pos = get_pos("бежал")
        assert pos in ('VERB', 'INFN')

    def test_get_pos_adjective(self):
        assert get_pos("красный") in ('ADJF', 'ADJS')

    def test_get_number_singular(self):
        assert get_number("дом") == 'sing'

    def test_get_number_plural(self):
        # "столы" однозначно множественное число
        assert get_number("столы") == 'plur'

    def test_get_gender_masculine(self):
        assert get_gender("красный") == 'masc'

    def test_get_gender_feminine(self):
        assert get_gender("красная") == 'femn'

    def test_get_gender_neuter(self):
        assert get_gender("красное") == 'neut'

    def test_get_case_nominative(self):
        case = get_case("дом")
        assert case == 'nomn'

    def test_get_case_genitive(self):
        case = get_case("дома")
        # "дома" может быть gent (sg) или nomn (pl)
        assert case in ('gent', 'nomn')


# =============================================================================
# ТЕСТЫ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ
# =============================================================================

@pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy2 not installed")
class TestAdditionalFunctions:

    def test_is_same_lemma_true(self):
        assert is_same_lemma("дом", "дома") is True

    def test_is_same_lemma_false(self):
        assert is_same_lemma("дом", "кот") is False

    def test_is_same_pos_true(self):
        assert is_same_pos("дом", "кот") is True  # оба NOUN

    def test_is_same_pos_false(self):
        assert is_same_pos("дом", "красный") is False  # NOUN vs ADJF

    def test_get_all_forms(self):
        forms = get_all_forms("дом")
        assert isinstance(forms, list)
        assert len(forms) > 1
        assert "дом" in forms

    def test_get_all_forms_empty_without_pymorphy(self):
        """Без pymorphy возвращает только исходное слово."""
        # Тест пройдёт если pymorphy есть — проверяем формат
        forms = get_all_forms("кот")
        assert "кот" in forms


# =============================================================================
# ТЕСТЫ КЭШИРОВАНИЯ
# =============================================================================

@pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy2 not installed")
class TestCaching:

    def test_parse_word_cached_consistent(self):
        """Кэш возвращает одинаковые результаты."""
        r1 = parse_word_cached("дом")
        r2 = parse_word_cached("дом")
        assert r1 == r2

    def test_cache_stats(self):
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert 'memory' in stats
        assert 'disk' in stats
        assert 'total_efficiency' in stats
        assert 'hits' in stats['memory']
        assert 'misses' in stats['memory']

    def test_disk_cache_singleton(self):
        c1 = get_disk_cache()
        c2 = get_disk_cache()
        assert c1 is c2

    def test_disk_cache_get_set(self):
        cache = get_disk_cache()
        test_word = "__test_word_12345__"
        test_info = ("тест", "NOUN", "sing", "masc", "nomn")
        cache.set(test_word, test_info)
        result = cache.get(test_word)
        assert result == test_info

    def test_disk_cache_stats(self):
        cache = get_disk_cache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'db_path' in stats


# =============================================================================
# ТЕСТЫ БЕЗ PYMORPHY
# =============================================================================

class TestWithoutPymorphy:
    """Тесты, работающие без pymorphy2."""

    def test_normalize_works(self):
        assert normalize_word("ТЕСТ") == "тест"

    def test_version_exists(self):
        assert VERSION is not None

    def test_has_pymorphy_flag(self):
        assert isinstance(HAS_PYMORPHY, bool)


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
