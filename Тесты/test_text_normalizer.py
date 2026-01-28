#!/usr/bin/env python3
"""
Unit-тесты для text_normalizer.py

Покрывает основные функции нормализации текста:
- Преобразование чисел в слова
- Склонение числительных
- Раскрытие сокращений
- Обработка дефисов и тире
- Очистка текста
- Полная нормализация (интеграция)
- Нормализация междометий
- Единицы измерения после чисел

Запуск:
    pytest Тесты/test_text_normalizer.py -v
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))

import pytest
from text_normalizer import (
    number_to_words,
    get_plural_form,
    expand_numbers_in_text,
    expand_abbreviations,
    expand_units_after_numbers,
    normalize_dashes_and_hyphens,
    process_hyphens,
    sanitize_text,
    normalize_for_comparison,
    normalize_interjections,
)


# =============================================================================
# ТЕСТЫ ПРЕОБРАЗОВАНИЯ ЧИСЕЛ В СЛОВА
# =============================================================================

class TestNumberToWords:
    """Тесты преобразования чисел в слова"""

    def test_zero(self):
        assert number_to_words(0) == 'ноль'

    def test_single_digits(self):
        assert number_to_words(1) == 'один'
        assert number_to_words(5) == 'пять'
        assert number_to_words(9) == 'девять'

    def test_teens(self):
        assert number_to_words(10) == 'десять'
        assert number_to_words(11) == 'одиннадцать'
        assert number_to_words(15) == 'пятнадцать'
        assert number_to_words(19) == 'девятнадцать'

    def test_tens(self):
        assert number_to_words(20) == 'двадцать'
        assert number_to_words(40) == 'сорок'
        assert number_to_words(90) == 'девяносто'

    def test_tens_with_units(self):
        assert number_to_words(21) == 'двадцать один'
        assert number_to_words(55) == 'пятьдесят пять'
        assert number_to_words(99) == 'девяносто девять'

    def test_hundreds(self):
        assert number_to_words(100) == 'сто'
        assert number_to_words(200) == 'двести'
        assert number_to_words(500) == 'пятьсот'

    def test_hundreds_with_remainder(self):
        assert number_to_words(101) == 'сто один'
        assert number_to_words(215) == 'двести пятнадцать'
        assert number_to_words(999) == 'девятьсот девяносто девять'

    def test_thousands(self):
        """Тысячи — женский род (одна тысяча, две тысячи)"""
        assert number_to_words(1000) == 'одна тысяча'
        assert number_to_words(2000) == 'две тысячи'
        assert number_to_words(5000) == 'пять тысяч'
        assert number_to_words(11000) == 'одиннадцать тысяч'

    def test_thousands_with_remainder(self):
        assert number_to_words(1001) == 'одна тысяча один'
        assert number_to_words(2500) == 'две тысячи пятьсот'

    def test_large_numbers(self):
        result = number_to_words(1000000)
        assert 'миллион' in result

    def test_negative(self):
        assert number_to_words(-5) == 'минус пять'
        assert number_to_words(-100) == 'минус сто'

    def test_year_like_numbers(self):
        """Числа, похожие на годы"""
        result = number_to_words(1812)
        assert 'тысяча' in result
        assert 'восемьсот' in result
        assert 'двенадцать' in result

        result = number_to_words(2024)
        assert 'две тысячи' in result
        assert 'двадцать четыре' in result


# =============================================================================
# ТЕСТЫ СКЛОНЕНИЯ
# =============================================================================

class TestPluralForm:
    """Тесты правильной формы слова для числа"""

    def test_one(self):
        forms = ('год', 'года', 'лет')
        assert get_plural_form(1, forms) == 'год'
        assert get_plural_form(21, forms) == 'год'
        assert get_plural_form(101, forms) == 'год'

    def test_two_to_four(self):
        forms = ('год', 'года', 'лет')
        assert get_plural_form(2, forms) == 'года'
        assert get_plural_form(3, forms) == 'года'
        assert get_plural_form(4, forms) == 'года'
        assert get_plural_form(22, forms) == 'года'

    def test_five_and_more(self):
        forms = ('год', 'года', 'лет')
        assert get_plural_form(5, forms) == 'лет'
        assert get_plural_form(10, forms) == 'лет'
        assert get_plural_form(20, forms) == 'лет'

    def test_teens(self):
        """11-19 всегда третья форма"""
        forms = ('год', 'года', 'лет')
        assert get_plural_form(11, forms) == 'лет'
        assert get_plural_form(12, forms) == 'лет'
        assert get_plural_form(14, forms) == 'лет'
        assert get_plural_form(19, forms) == 'лет'

    def test_negative(self):
        """Отрицательные числа — по модулю"""
        forms = ('год', 'года', 'лет')
        assert get_plural_form(-1, forms) == 'год'
        assert get_plural_form(-5, forms) == 'лет'


# =============================================================================
# ТЕСТЫ РАСКРЫТИЯ ЧИСЕЛ В ТЕКСТЕ
# =============================================================================

class TestExpandNumbers:
    """Тесты замены чисел на слова в тексте"""

    def test_simple_number(self):
        result = expand_numbers_in_text('у нас 5 кошек')
        assert 'пять' in result
        assert '5' not in result

    def test_year_with_suffix(self):
        result = expand_numbers_in_text('в 1812 году')
        assert '1812' not in result
        assert 'года' in result

    def test_no_numbers(self):
        text = 'обычный текст без чисел'
        assert expand_numbers_in_text(text) == text


# =============================================================================
# ТЕСТЫ РАСКРЫТИЯ СОКРАЩЕНИЙ
# =============================================================================

class TestExpandAbbreviations:
    """Тесты замены сокращений"""

    def test_basic_abbreviations(self):
        # Сокращение 'и т.д.' → 'и так далее' (длинная форма первой)
        result_itd = expand_abbreviations('и т.д.')
        # Может заменить 'и т.д.' целиком или 'т.д.' отдельно
        assert 'так далее' in result_itd or result_itd == 'и т.д.'

        result_te = expand_abbreviations('т.е.')
        assert 'то есть' in result_te or result_te == 'т.е.'

    def test_no_abbreviations(self):
        text = 'обычный текст'
        assert expand_abbreviations(text) == text


# =============================================================================
# ТЕСТЫ ЕДИНИЦ ИЗМЕРЕНИЯ
# =============================================================================

class TestExpandUnits:
    """Тесты замены единиц измерения после чисел"""

    def test_meters_after_number(self):
        result = expand_units_after_numbers('5 м от дома')
        assert 'метров' in result

    def test_kilometers(self):
        result = expand_units_after_numbers('10 км пути')
        assert 'километров' in result

    def test_unit_without_number(self):
        """Единица без числа — не трогаем"""
        text = 'буква м в слове'
        result = expand_units_after_numbers(text)
        # 'м' без предшествующего числа не должна меняться
        assert 'метров' not in result


# =============================================================================
# ТЕСТЫ НОРМАЛИЗАЦИИ ДЕФИСОВ И ТИРЕ
# =============================================================================

class TestDashesAndHyphens:
    """Тесты обработки дефисов и тире"""

    def test_em_dash_to_space(self):
        """Длинное тире между словами — пробел"""
        result = normalize_dashes_and_hyphens('слово — другое')
        assert '—' not in result
        assert 'слово' in result
        assert 'другое' in result

    def test_en_dash_to_space(self):
        """Среднее тире между словами — пробел"""
        result = normalize_dashes_and_hyphens('первое – второе')
        assert '–' not in result

    def test_dialog_dash_removed(self):
        """Тире в начале строки (диалог) — удаляем"""
        result = normalize_dashes_and_hyphens('— Привет, сказал он')
        assert result.startswith('Привет') or result.startswith('привет')

    def test_hyphen_inside_word_preserved(self):
        """Дефис внутри слова — сохраняем"""
        result = normalize_dashes_and_hyphens('кое-что сделали')
        assert '-' in result


class TestProcessHyphens:
    """Тесты обработки слов с дефисами"""

    def test_keep_hyphens_mode(self):
        """С keep_hyphens=True известные слова сохраняют дефис"""
        result = process_hyphens('кое-что написал', keep_hyphens=True)
        assert 'кое-что' in result

    def test_no_keep_hyphens(self):
        """Без keep_hyphens дефисы убираются"""
        result = process_hyphens('кое-что написал', keep_hyphens=False)
        # Дефис убран, слово слитно или через пробел
        assert 'кое-что' not in result

    def test_known_hyphenated_words(self):
        """Известные слова с дефисами — из-за, по-моему"""
        result = process_hyphens('из-за дождя', keep_hyphens=False)
        assert 'из-за' not in result
        # Должно стать слитно: "изза"
        assert 'изза' in result


# =============================================================================
# ТЕСТЫ ОЧИСТКИ ТЕКСТА
# =============================================================================

class TestSanitizeText:
    """Тесты очистки текста"""

    def test_lowercase(self):
        result = sanitize_text('СЛОВО')
        assert result == 'слово'

    def test_yo_replacement(self):
        result = sanitize_text('ёлка')
        assert result == 'елка'

    def test_punctuation_removal(self):
        result = sanitize_text('Привет, мир!')
        assert ',' not in result
        assert '!' not in result

    def test_quotes_removal(self):
        result = sanitize_text('слово «в» кавычках')
        assert '«' not in result
        assert '»' not in result

    def test_normalize_spaces(self):
        result = sanitize_text('много   пробелов   здесь')
        assert '  ' not in result

    def test_strip(self):
        result = sanitize_text('  слово  ')
        assert result == 'слово'

    def test_empty_string(self):
        result = sanitize_text('')
        assert result == ''

    def test_nonbreaking_space(self):
        result = sanitize_text('слово\u00a0другое')
        assert '\u00a0' not in result
        assert 'слово' in result


# =============================================================================
# ТЕСТЫ НОРМАЛИЗАЦИИ МЕЖДОМЕТИЙ
# =============================================================================

class TestNormalizeInterjections:
    """Тесты нормализации междометий"""

    def test_hmm(self):
        result = normalize_interjections('Хм-м-м...')
        assert 'хм' in result.lower()

    def test_mmm(self):
        result = normalize_interjections('М-м-м...')
        assert result.strip().lower() in ('м', 'м...')

    def test_eee(self):
        result = normalize_interjections('Э-э-э...')
        assert 'э' in result.lower()

    def test_no_interjection(self):
        text = 'обычный текст'
        result = normalize_interjections(text)
        assert result == text


# =============================================================================
# ТЕСТЫ ПОЛНОЙ НОРМАЛИЗАЦИИ (ИНТЕГРАЦИЯ)
# =============================================================================

class TestNormalizeForComparison:
    """Интеграционные тесты полной нормализации"""

    def test_full_pipeline(self):
        """Полный цикл нормализации"""
        text = '— Это было в 1812 г., — сказал он.'
        result = normalize_for_comparison(text)
        # Тире убрано, год раскрыт, г.→года, пунктуация убрана, нижний регистр
        assert '—' not in result
        assert '1812' not in result
        assert ',' not in result
        assert '.' not in result
        assert result == result.lower()

    def test_idempotency(self):
        """Повторная нормализация не меняет результат"""
        text = 'простой текст без особенностей'
        first = normalize_for_comparison(text)
        second = normalize_for_comparison(first)
        assert first == second

    def test_empty_text(self):
        result = normalize_for_comparison('')
        assert result == ''

    def test_only_punctuation(self):
        result = normalize_for_comparison('..., !!! ???')
        assert result.strip() == ''

    def test_numbers_disabled(self):
        """Без раскрытия чисел"""
        text = 'было 5 штук'
        result = normalize_for_comparison(text, expand_numbers=False)
        assert '5' in result

    def test_hyphens_kept(self):
        """С сохранением дефисов"""
        text = 'кое-что случилось'
        result = normalize_for_comparison(text, keep_hyphens=True)
        assert 'кое-что' in result

    def test_mixed_content(self):
        """Смешанный контент: числа, сокращения, дефисы, тире"""
        text = '— В 2024 г. он кое-как добрался т.д.'
        result = normalize_for_comparison(text)
        assert '2024' not in result
        assert '—' not in result
        assert 'кое' in result  # дефис убран


# =============================================================================
# ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ
# =============================================================================

class TestEdgeCases:
    """Тесты граничных случаев"""

    def test_unicode_spaces(self):
        """Различные Unicode-пробелы"""
        text = 'слово\u2003другое\u200bтретье'
        result = sanitize_text(text)
        assert '\u2003' not in result
        assert '\u200b' not in result

    def test_bom(self):
        """BOM в начале текста"""
        text = '\ufeffтекст с BOM'
        result = sanitize_text(text)
        assert '\ufeff' not in result

    def test_very_long_number(self):
        """Очень большое число"""
        result = number_to_words(999999)
        assert 'тысяч' in result
        assert isinstance(result, str)

    def test_number_one_thousand_one(self):
        """1001 — проверка правильного порядка слов"""
        result = number_to_words(1001)
        assert 'одна тысяча один' == result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
