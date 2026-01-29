#!/usr/bin/env python3
"""
Text Normalizer v2.0 - Нормализация текста для сверки с транскрипцией

Приводит исходный текст книги к виду, максимально близкому к тому,
что распознает Яндекс SpeechKit.

Этапы нормализации:
1. Текстуализация чисел и сокращений (через правила)
2. Обработка слов с дефисами
3. Очистка (пунктуация, регистр, ё→е)

Использование:
    python text_normalizer.py текст.txt --output нормализованный.txt
    python text_normalizer.py текст.txt --keep-hyphens  # сохранить дефисы
"""

import argparse
import json
import os
import re
from pathlib import Path

# Импорт централизованной конфигурации
try:
    from config import (
        FileNaming, CHAPTERS_DIR, check_file_exists
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


# =============================================================================
# ТЕКСТУАЛИЗАЦИЯ ЧИСЕЛ
# =============================================================================

# Единицы
ONES = {
    0: '', 1: 'один', 2: 'два', 3: 'три', 4: 'четыре',
    5: 'пять', 6: 'шесть', 7: 'семь', 8: 'восемь', 9: 'девять',
    10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать',
    14: 'четырнадцать', 15: 'пятнадцать', 16: 'шестнадцать',
    17: 'семнадцать', 18: 'восемнадцать', 19: 'девятнадцать',
}

# Десятки
TENS = {
    2: 'двадцать', 3: 'тридцать', 4: 'сорок', 5: 'пятьдесят',
    6: 'шестьдесят', 7: 'семьдесят', 8: 'восемьдесят', 9: 'девяносто',
}

# Сотни
HUNDREDS = {
    1: 'сто', 2: 'двести', 3: 'триста', 4: 'четыреста',
    5: 'пятьсот', 6: 'шестьсот', 7: 'семьсот', 8: 'восемьсот', 9: 'девятьсот',
}

# Порядки
THOUSANDS = ['', 'тысяча', 'миллион', 'миллиард', 'триллион']
THOUSANDS_FORMS = {
    'тысяча': ('тысяча', 'тысячи', 'тысяч'),
    'миллион': ('миллион', 'миллиона', 'миллионов'),
    'миллиард': ('миллиард', 'миллиарда', 'миллиардов'),
    'триллион': ('триллион', 'триллиона', 'триллионов'),
}

# Сокращения
ABBREVIATIONS = {
    'г.': 'года',
    'гг.': 'годов',
    'в.': 'века',
    'вв.': 'веков',
    'руб': 'рублей',
    'коп': 'копеек',
    'тыс': 'тысяч',
    'млн': 'миллионов',
    'млрд': 'миллиардов',
    'др': 'другое',
    'т.д.': 'так далее',
    'т.п.': 'тому подобное',
    'т.е.': 'то есть',
    'т.к.': 'так как',
    'т.н.': 'так называемый',
    'и т.д.': 'и так далее',
    'и т.п.': 'и тому подобное',
    'и др.': 'и другие',
    'и пр.': 'и прочее',
    'см.': 'смотри',
    'ср.': 'сравни',
    'напр.': 'например',
    'прим.': 'примечание',
    'гл.': 'глава',
    'стр.': 'страница',
    'с.': 'страница',
    'ок.': 'около',
    'прибл.': 'приблизительно',
}

# Единицы измерения — заменяются ТОЛЬКО после чисел
# Например: "5 м" → "5 метров", но "м…" (междометие) не трогаем
# Примечание: 'г' не включён — конфликт с "года" (обрабатывается в expand_numbers_in_text)
UNITS_OF_MEASURE = {
    'л': 'литров',
    'м': 'метров',
    'км': 'километров',
    'кг': 'килограммов',
    'см': 'сантиметров',
    'мм': 'миллиметров',
    'мл': 'миллилитров',
}


def expand_units_after_numbers(text):
    """
    Заменяет единицы измерения на полные формы ТОЛЬКО после чисел.
    Например: "5 м" → "5 метров", но одиночное "м" не трогаем.
    """
    for unit, expansion in sorted(UNITS_OF_MEASURE.items(), key=lambda x: -len(x[0])):
        # Паттерн: число + пробел(ы) + единица измерения
        pattern = rf'(\d+)\s*{re.escape(unit)}\b'
        text = re.sub(pattern, rf'\1 {expansion}', text, flags=re.IGNORECASE)
    return text


def get_plural_form(n, forms):
    """
    Возвращает правильную форму слова для числа.
    forms = (один, два, пять) — например ('год', 'года', 'лет')
    """
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n = n % 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def number_to_words(n, feminine=False):
    """
    Преобразует число в слова.
    feminine=True для женского рода (одна тысяча, две тысячи)
    """
    if n == 0:
        return 'ноль'

    if n < 0:
        return 'минус ' + number_to_words(-n, feminine)

    result = []

    # Разбиваем на группы по 3 цифры
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000

    for i, group in enumerate(groups):
        if group == 0:
            continue

        group_words = []

        # Сотни
        h = group // 100
        if h > 0:
            group_words.append(HUNDREDS[h])

        # Десятки и единицы
        remainder = group % 100
        if remainder > 0:
            if remainder < 20:
                word = ONES[remainder]
                # Для тысяч используем женский род
                if i == 1:  # тысячи
                    if remainder == 1:
                        word = 'одна'
                    elif remainder == 2:
                        word = 'две'
                elif feminine and remainder in (1, 2):
                    if remainder == 1:
                        word = 'одна'
                    elif remainder == 2:
                        word = 'две'
                group_words.append(word)
            else:
                t = remainder // 10
                o = remainder % 10
                group_words.append(TENS[t])
                if o > 0:
                    word = ONES[o]
                    if i == 1 and o in (1, 2):
                        word = 'одна' if o == 1 else 'две'
                    group_words.append(word)

        # Добавляем порядок (тысяча, миллион, ...)
        if i > 0 and THOUSANDS[i]:
            order = THOUSANDS[i]
            if order in THOUSANDS_FORMS:
                # Определяем форму по последним цифрам группы
                last_digits = group % 100
                if last_digits >= 11 and last_digits <= 19:
                    order = THOUSANDS_FORMS[order][2]
                else:
                    last_digit = group % 10
                    if last_digit == 1:
                        order = THOUSANDS_FORMS[order][0]
                    elif 2 <= last_digit <= 4:
                        order = THOUSANDS_FORMS[order][1]
                    else:
                        order = THOUSANDS_FORMS[order][2]
            group_words.append(order)

        result = group_words + result

    return ' '.join(filter(None, result))


def expand_year(year_str):
    """Преобразует год в слова"""
    try:
        year = int(year_str)
        if 1000 <= year <= 2100:
            return number_to_words(year) + ' год'
        return number_to_words(year)
    except ValueError:
        return year_str


def expand_numbers_in_text(text):
    """
    Заменяет числа в тексте на слова.
    Обрабатывает годы, даты, обычные числа.
    """
    # Годы (1812, 2024 и т.д.)
    text = re.sub(
        r'\b(1[0-9]{3}|20[0-2][0-9])\s*г(?:ода?|\.)?',
        lambda m: number_to_words(int(m.group(1))) + ' года',
        text
    )

    # Века (XIX, XX, XXI)
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    def roman_to_int(s):
        result = 0
        prev = 0
        for c in reversed(s.upper()):
            curr = roman_map.get(c, 0)
            if curr < prev:
                result -= curr
            else:
                result += curr
            prev = curr
        return result

    def expand_century(m):
        roman = m.group(1)
        century = roman_to_int(roman)
        return number_to_words(century) + ' век'

    text = re.sub(r'\b([IVXLC]+)\s*в(?:ека?|\.)?', expand_century, text)

    # Числа с единицами измерения (полные формы для склонения)
    # Примечание: основной список UNITS_OF_MEASURE выше — для простой замены
    units_with_declension = {
        'км': 'километр', 'м': 'метр', 'см': 'сантиметр', 'мм': 'миллиметр',
        'кг': 'килограмм', 'г': 'грамм', 'л': 'литр', 'мл': 'миллилитр',
        'руб': 'рубль', 'коп': 'копейка',
    }

    for abbr, word in units_with_declension.items():
        pattern = rf'\b(\d+)\s*{re.escape(abbr)}\.?(?!\w)'
        def make_replacement(w):
            def replacement(m):
                n = int(m.group(1))
                # Простое склонение
                if w.endswith('р'):  # метр, литр
                    forms = (w, w + 'а', w + 'ов')
                elif w.endswith('м'):  # килограмм, грамм
                    forms = (w, w + 'а', w + 'ов')
                elif w.endswith('ь'):  # рубль
                    base = w[:-1]
                    forms = (w, base + 'я', base + 'ей')
                else:
                    forms = (w, w + 'ы', w)
                return number_to_words(n) + ' ' + get_plural_form(n, forms)
            return replacement
        text = re.sub(pattern, make_replacement(word), text, flags=re.IGNORECASE)

    # Обычные числа
    text = re.sub(r'\b(\d+)\b', lambda m: number_to_words(int(m.group(1))), text)

    return text


def expand_abbreviations(text):
    """Заменяет сокращения на полные формы"""
    # Сортируем по длине (длинные сначала)
    for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        # Учитываем границы слов и регистр
        pattern = re.escape(abbr)
        text = re.sub(rf'\b{pattern}\b', expansion, text, flags=re.IGNORECASE)
    return text


# =============================================================================
# ОБРАБОТКА МЕЖДОМЕТИЙ И ЗАМИНОК
# =============================================================================

# Паттерны междометий с повторами (Хм-м-м, М-м, Кх-м, Э-э-э и т.д.)
# Яндекс их не распознаёт, поэтому нормализуем до базовой формы
INTERJECTION_PATTERNS = [
    # Хм-м-м... → хм
    (r'\b[Хх]м[-–—]?м[-–—]?м*\.{0,3}', 'хм'),
    (r'\b[Хх][-–—]м[-–—]?м*\.{0,3}', 'хм'),
    # Кх-м → кхм
    (r'\b[Кк]х[-–—]м\.{0,3}', 'кхм'),
    # М-м-м... → м
    (r'\b[Мм][-–—]м[-–—]?м*\.{0,3}', 'м'),
    # Э-э-э... → э
    (r'\b[Ээ][-–—]э[-–—]?э*\.{0,3}', 'э'),
    # А-а-а... → а (протяжное)
    (r'\b[Аа][-–—]а[-–—]?а*\.{0,3}', 'а'),
    # У-у-у... → у
    (r'\b[Уу][-–—]у[-–—]?у*\.{0,3}', 'у'),
    # О-о-о... → о
    (r'\b[Оо][-–—]о[-–—]?о*\.{0,3}', 'о'),
    # Н-да, Н-ну → нда, нну
    (r'\b[Нн][-–—]да\b', 'нда'),
    (r'\b[Нн][-–—]ну\b', 'нну'),
    # Одиночные буквы с многоточием (М…, Э…) — оборванные слова/заминки
    # Обрабатываем в конце, чтобы не мешать более специфичным паттернам
    (r'\b[Мм]…', 'м'),
    (r'\b[Ээ]…', 'э'),
    (r'\b[Аа]…', 'а'),
]

# Компилируем паттерны
INTERJECTION_PATTERNS_COMPILED = [(re.compile(p), r) for p, r in INTERJECTION_PATTERNS]


def normalize_interjections(text):
    """
    Нормализует междометия и заминки в речи.

    Примеры:
        "Хм-м-м…" → "хм"
        "Кх-м, выгреб" → "кхм выгреб"
        "М-м-м — Зеленорукий" → "м Зеленорукий"
    """
    result = text
    for pattern, replacement in INTERJECTION_PATTERNS_COMPILED:
        result = pattern.sub(replacement, result)
    return result


# =============================================================================
# ОБРАБОТКА ДЕФИСОВ И ТИРЕ
# =============================================================================

# Все варианты дефисов и тире
HYPHEN_CHARS = {
    '-',      # обычный дефис-минус (U+002D)
    '‐',      # дефис (U+2010)
    '‑',      # неразрывный дефис (U+2011)
    '‒',      # цифровое тире (U+2012)
    '–',      # среднее тире (U+2013)
    '—',      # длинное тире (U+2014)
    '―',      # горизонтальная черта (U+2015)
    '−',      # знак минуса (U+2212)
}

# Стандартный дефис для унификации
STANDARD_HYPHEN = '-'

# Слова с дефисами, которые нужно СОХРАНИТЬ (дефис внутри слова)
# Эти слова произносятся слитно, дефис остаётся
KEEP_HYPHENATED = {
    # кое-
    'кое-как', 'кое-где', 'кое-кто', 'кое-что', 'кое-куда', 'кое-какой',
    # -то, -нибудь, -либо
    'как-то', 'как-нибудь', 'как-либо',
    'когда-то', 'когда-нибудь', 'когда-либо',
    'где-то', 'где-нибудь', 'где-либо',
    'куда-то', 'куда-нибудь', 'куда-либо',
    'кто-то', 'кто-нибудь', 'кто-либо',
    'что-то', 'что-нибудь', 'что-либо',
    'какой-то', 'какой-нибудь', 'какой-либо',
    'какая-то', 'какая-нибудь', 'какая-либо',
    'какое-то', 'какое-нибудь', 'какое-либо',
    'какие-то', 'какие-нибудь', 'какие-либо',
    'чей-то', 'чей-нибудь', 'чей-либо',
    'почему-то', 'отчего-то', 'зачем-то', 'откуда-то',
    # по-
    'по-моему', 'по-твоему', 'по-своему', 'по-нашему', 'по-вашему',
    'по-русски', 'по-английски', 'по-немецки', 'по-французски', 'по-латыни',
    'по-прежнему', 'по-видимому', 'по-настоящему', 'по-разному', 'по-другому',
    'по-новому', 'по-старому', 'по-хорошему', 'по-плохому', 'по-доброму',
    'по-человечески', 'по-братски', 'по-дружески',
    # во-, в-
    'во-первых', 'во-вторых', 'в-третьих', 'в-четвёртых', 'в-пятых',
    # из-
    'из-за', 'из-под',
    # Устойчивые выражения
    'всё-таки', 'все-таки', 'так-таки',
    'вот-вот', 'еле-еле', 'чуть-чуть', 'едва-едва', 'только-только',
    'мало-помалу', 'давным-давно', 'видимо-невидимо',
    'нежданно-негаданно', 'подобру-поздорову',
    'худо-бедно', 'шиворот-навыворот',
    # Дополнительные
    'всё-ещё', 'все-ещё', 'ещё-бы',
    'по-всякому', 'по-любому', 'по-иному',
    'в-шестых', 'в-седьмых', 'в-восьмых', 'в-девятых', 'в-десятых',
    'точь-в-точь', 'один-единственный', 'одна-единственная',
    'крест-накрест', 'туда-сюда', 'там-сям',
}


def normalize_dashes_and_hyphens(text):
    """
    Нормализация дефисов и тире по правилам:

    1. Дефисы ВНУТРИ слов → унифицируем до стандартного дефиса (-)
    2. Тире МЕЖДУ словами → заменяем на пробел
    3. Тире в начале строки (диалоги) → удаляем

    Стратегия: "Унифицируй или разделяй"
    """
    result = text

    # ШАГ 1: Тире в начале строки (диалоги) → удаляем
    # Паттерн: начало строки + любое тире + возможный пробел
    result = re.sub(r'^[\-\u2010-\u2015\u2212]+\s*', '', result, flags=re.MULTILINE)

    # ШАГ 2: Тире МЕЖДУ словами → пробел
    # Паттерн: пробел/начало + тире + пробел/конец
    # Длинное и среднее тире (— –) обычно между словами
    result = re.sub(r'\s*[—–―]\s*', ' ', result)

    # ШАГ 3: Дефисы ВНУТРИ слов → унифицируем
    # Паттерн: буква + любой дефис + буква
    def unify_hyphen(match):
        return match.group(1) + STANDARD_HYPHEN + match.group(2)

    hyphen_pattern = r'(\w)[' + re.escape(''.join(HYPHEN_CHARS)) + r'](\w)'
    result = re.sub(hyphen_pattern, unify_hyphen, result)

    return result


def process_hyphens(text, keep_hyphens=False):
    """
    Обрабатывает слова с дефисами для сравнения с транскрипцией.

    Если keep_hyphens=True:
        - Сохраняем дефисы в известных словах (кое-что, по-моему)
        - Остальные дефисы → пробелы

    Если keep_hyphens=False:
        - Все дефисы удаляем или заменяем на пробелы
    """
    # Сначала нормализуем все дефисы и тире
    result = normalize_dashes_and_hyphens(text)

    if keep_hyphens:
        # Сохраняем дефисы в известных словах, остальные → пробелы
        # Сначала защищаем известные слова
        protected = {}
        for i, word in enumerate(KEEP_HYPHENATED):
            placeholder = f'__HYPHEN_{i}__'
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            if pattern.search(result):
                protected[placeholder] = word
                result = pattern.sub(placeholder, result)

        # Заменяем оставшиеся дефисы на пробелы
        result = re.sub(r'(\w)-(\w)', r'\1 \2', result)

        # Восстанавливаем защищённые слова
        for placeholder, word in protected.items():
            result = result.replace(placeholder, word)
    else:
        # Все дефисы → убираем или пробелы

        # Сначала обрабатываем известные слова (убираем дефис)
        for word in KEEP_HYPHENATED:
            merged = word.replace('-', '')
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            result = pattern.sub(merged, result)

        # Повторы (синий-синий → синий синий)
        result = re.sub(r'\b(\w+)-\1\b', r'\1 \1', result)

        # Оставшиеся дефисы → пробелы
        result = re.sub(r'(\w)-(\w)', r'\1 \2', result)

    return result


# =============================================================================
# ОЧИСТКА ТЕКСТА
# =============================================================================

def sanitize_text(text, keep_case=False):
    """
    Очистка текста для сравнения с транскрипцией.

    Порядок важен:
    1. Нормализация дефисов и тире (до удаления пунктуации!)
    2. ё → е
    3. Нижний регистр
    4. Удаление пунктуации (кроме уже обработанных дефисов)
    5. Нормализация пробелов
    """
    # ВАЖНО: Дефисы и тире обрабатываем ДО удаления пунктуации
    # Это уже сделано в process_hyphens(), здесь только типографика

    # ё → е
    text = text.replace('ё', 'е').replace('Ё', 'Е')

    # Нижний регистр
    if not keep_case:
        text = text.lower()

    # Типографские символы → обычные/пробелы
    # НЕ трогаем дефисы здесь - они обработаны в process_hyphens()
    replacements = {
        # Кавычки → удаляем
        '«': '',
        '»': '',
        '"': '',
        '"': '',
        '"': '',
        '„': '',
        ''': '',
        ''': '',
        "'": '',
        # Многоточие
        '…': '',
        # Пробелы разных видов → обычный пробел
        '\u00a0': ' ',  # неразрывный пробел
        '\u200b': '',   # zero-width space
        '\u2009': ' ',  # thin space
        '\u2002': ' ',  # en space
        '\u2003': ' ',  # em space
        '\u202f': ' ',  # narrow no-break space
        '\ufeff': '',   # BOM
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Удаляем пунктуацию КРОМЕ дефиса внутри слов
    # \w = буквы и цифры, \s = пробелы, - = дефис
    # Удаляем всё остальное
    text = re.sub(r'[^\w\s\-]', ' ', text)

    # Но дефисы на границах слов (не внутри) → пробелы
    # Например: "слово- " или " -слово"
    text = re.sub(r'\s+-', ' ', text)
    text = re.sub(r'-\s+', ' ', text)

    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def normalize_for_comparison(text, expand_numbers=True, expand_abbrev=True,
                              process_hyphen=True, keep_hyphens=False):
    """
    Полная нормализация текста для сравнения с транскрипцией.

    Args:
        text: исходный текст
        expand_numbers: заменять числа на слова
        expand_abbrev: заменять сокращения
        process_hyphen: обрабатывать дефисы
        keep_hyphens: сохранять дефисы (иначе убираем/заменяем на пробел)

    Returns:
        нормализованный текст
    """
    result = text

    # 0. Междометия и заминки (ДО дефисов, т.к. "Хм-м-м" содержит дефисы)
    result = normalize_interjections(result)

    # 1. Сокращения (до чисел, т.к. могут быть "2 г." → "2 года")
    if expand_abbrev:
        result = expand_abbreviations(result)
        # Единицы измерения только после чисел (5 м → 5 метров)
        result = expand_units_after_numbers(result)

    # 2. Числа
    if expand_numbers:
        result = expand_numbers_in_text(result)

    # 3. Дефисы
    if process_hyphen:
        result = process_hyphens(result, keep_hyphens)

    # 4. Очистка
    result = sanitize_text(result)

    return result


def normalize_file(input_path, output_path=None, force=False, **kwargs):
    """
    Нормализует текст из файла.

    Args:
        input_path: путь к входному файлу (TXT или DOCX)
        output_path: путь для сохранения (автоматически если не указан)
        force: перезаписать существующий файл без предупреждения
        **kwargs: параметры для normalize_for_comparison()

    Returns:
        нормализованный текст
    """
    input_path = Path(input_path)

    # Читаем
    if input_path.suffix.lower() == '.docx':
        try:
            from docx import Document
            doc = Document(str(input_path))
            text = '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError("Для DOCX установите: pip install python-docx")
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

    print(f"Загружен: {input_path.name}")
    print(f"  Символов: {len(text)}")
    print(f"  Слов: {len(text.split())}")

    # Нормализуем
    normalized = normalize_for_comparison(text, **kwargs)

    print(f"\nПосле нормализации:")
    print(f"  Символов: {len(normalized)}")
    print(f"  Слов: {len(normalized.split())}")

    # Определяем выходной путь
    if output_path:
        out_file = Path(output_path)
    else:
        # Используем FileNaming для правильного имени
        if HAS_CONFIG:
            chapter_id = FileNaming.get_chapter_id(input_path)
            out_file = input_path.parent / FileNaming.build_filename(chapter_id, 'normalized')
        else:
            out_file = input_path.with_stem(input_path.stem + '_normalized').with_suffix('.txt')

    # Проверяем существование файла
    if out_file.exists() and not force:
        if HAS_CONFIG:
            check_file_exists(out_file, action='ask')
        else:
            print(f"  ⚠ Файл уже существует: {out_file.name}")

    # Сохраняем
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(normalized)

    print(f"\nСохранено: {out_file}")

    return normalized


def main():
    parser = argparse.ArgumentParser(
        description='Нормализация текста для сравнения с транскрипцией',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Этапы нормализации:
  1. Замена сокращений (г. → года, т.д. → так далее)
  2. Замена чисел на слова (2024 → две тысячи двадцать четыре)
  3. Обработка дефисов (кое-что → коечто, диван-кровать → диван кровать)
  4. Очистка (пунктуация, регистр, ё→е)

Примеры:
  python text_normalizer.py глава.txt
  python text_normalizer.py глава.docx --output готово.txt
  python text_normalizer.py глава.txt --keep-hyphens
        """
    )
    parser.add_argument('input', help='Входной файл (TXT или DOCX)')
    parser.add_argument('--output', '-o', help='Выходной файл')
    parser.add_argument('--no-numbers', action='store_true',
                        help='Не заменять числа на слова')
    parser.add_argument('--no-abbrev', action='store_true',
                        help='Не заменять сокращения')
    parser.add_argument('--keep-hyphens', action='store_true',
                        help='Сохранять дефисы в словах')
    parser.add_argument('--force', action='store_true',
                        help='Перезаписать существующие файлы')

    args = parser.parse_args()

    try:
        normalize_file(
            args.input,
            output_path=args.output,
            force=args.force,
            expand_numbers=not args.no_numbers,
            expand_abbrev=not args.no_abbrev,
            keep_hyphens=args.keep_hyphens
        )
        print("\n✓ Готово!")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
