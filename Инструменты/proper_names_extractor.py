#!/usr/bin/env python3
"""
Proper Names Extractor v2.0 - Автоизвлечение имён собственных из DOCX

Анализирует текст и извлекает потенциальные имена собственные
для добавления в защищённые слова.

Методы определения:
1. Слова с заглавной буквы не в начале предложения
2. Морфологический анализ (pymorphy2: Name, Surn, Patr)
3. Паттерны имён (Имя Отчество, Имя Фамилия)
4. Исключение стоп-слов (месяцы, дни недели и т.д.)

Использование:
    from proper_names_extractor import extract_names_from_docx, extract_names_from_text

    names = extract_names_from_docx("глава.docx")
    print(names)  # {'Иван', 'Петрович', 'Москва'}

    # CLI: сохранить в словарь имён персонажей
    python proper_names_extractor.py глава.docx --to-names-dict
    python proper_names_extractor.py глава.docx --force

Changelog:
    v2.0 (2026-01-24): Интеграция с config.py
        - NAMES_DICT как путь по умолчанию для --protected
        - DICTIONARIES_DIR для выходных файлов
        - FileNaming для имени выходного файла
        - morphology.py вместо прямого импорта pymorphy2
        - check_file_exists() + флаг --force
        - VERSION/VERSION_DATE константы
    v1.0: Базовая версия извлечения имён
"""

# Версия модуля
VERSION = '5.0.0'
VERSION_DATE = '2026-01-25'

import re
from pathlib import Path
from typing import Set, List, Optional

# =============================================================================
# ИМПОРТ ЦЕНТРАЛИЗОВАННОЙ КОНФИГУРАЦИИ
# =============================================================================

try:
    from config import (
        NAMES_DICT, DICTIONARIES_DIR, FileNaming, check_file_exists
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Fallback пути
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    DICTIONARIES_DIR = PROJECT_DIR / 'Словари'
    NAMES_DICT = DICTIONARIES_DIR / 'Словарь_имён_персонажей.txt'

    def check_file_exists(path, action='skip'):
        """Fallback проверка существования файла."""
        if not path.exists():
            return True
        if action == 'overwrite':
            return True
        print(f"  → Файл уже существует: {path.name}")
        return action != 'skip'

# Импорт морфологии из централизованного модуля
try:
    from morphology import HAS_PYMORPHY, morph
except ImportError:
    # Fallback: прямой импорт pymorphy2
    try:
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        HAS_PYMORPHY = True
    except ImportError:
        morph = None
        HAS_PYMORPHY = False

# Попытка импортировать python-docx
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    Document = None
    HAS_DOCX = False


# =============================================================================
# СТОП-СЛОВА (не считаются именами собственными)
# =============================================================================

# Месяцы
MONTHS = {
    'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
    'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь',
    'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
}

# Дни недели
WEEKDAYS = {
    'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье',
}

# Частые слова, которые могут быть с заглавной в начале предложения
COMMON_WORDS = {
    'он', 'она', 'оно', 'они', 'мы', 'вы', 'ты', 'я',
    'это', 'тот', 'та', 'те', 'то', 'этот', 'эта', 'эти',
    'который', 'которая', 'которое', 'которые',
    'какой', 'какая', 'какое', 'какие',
    'весь', 'вся', 'всё', 'все',
    'сам', 'сама', 'само', 'сами',
    'свой', 'своя', 'своё', 'свои',
    'наш', 'наша', 'наше', 'наши',
    'ваш', 'ваша', 'ваше', 'ваши',
    'мой', 'моя', 'моё', 'мои',
    'твой', 'твоя', 'твоё', 'твои',
    'его', 'её', 'их',
    'кто', 'что', 'где', 'когда', 'как', 'почему', 'зачем',
    'да', 'нет', 'не', 'ни', 'но', 'и', 'а', 'или', 'ли',
    'бы', 'же', 'вот', 'вон', 'уж', 'ведь', 'даже',
    'здесь', 'там', 'тут', 'туда', 'сюда', 'оттуда', 'отсюда',
    'вдруг', 'теперь', 'потом', 'затем', 'сначала', 'наконец',
    'только', 'лишь', 'ещё', 'уже', 'снова', 'опять',
    'очень', 'совсем', 'почти', 'чуть', 'едва', 'слишком',
    'может', 'должен', 'нужно', 'надо', 'можно', 'нельзя',
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет',
    'иметь', 'имеет', 'имел', 'имела',
    'сказать', 'сказал', 'сказала', 'говорить', 'говорил', 'говорила',
    'думать', 'думал', 'думала', 'знать', 'знал', 'знала',
    'видеть', 'видел', 'видела', 'слышать', 'слышал', 'слышала',
    'хотеть', 'хотел', 'хотела', 'мочь', 'мог', 'могла',
    'после', 'перед', 'между', 'около', 'возле', 'вокруг',
    'через', 'сквозь', 'против', 'вдоль', 'поперёк',
    'над', 'под', 'за', 'перед', 'между',
    'один', 'одна', 'одно', 'одни', 'два', 'три', 'четыре', 'пять',
    'первый', 'второй', 'третий', 'четвёртый', 'пятый',
    'глава', 'часть', 'раздел', 'параграф',
    'страница', 'книга', 'текст', 'слово',
    'человек', 'люди', 'мужчина', 'женщина', 'ребёнок', 'дети',
    'господин', 'госпожа', 'товарищ',
    'утро', 'день', 'вечер', 'ночь', 'время', 'год', 'месяц', 'неделя',
}

# Объединяем все стоп-слова
STOP_WORDS = MONTHS | WEEKDAYS | COMMON_WORDS


# =============================================================================
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ
# =============================================================================

def is_proper_name_by_morph(word: str) -> bool:
    """
    Проверяет, является ли слово именем собственным через pymorphy2.

    Ищет теги: Name (имя), Surn (фамилия), Patr (отчество), Geox (географ.)
    """
    if not HAS_PYMORPHY or not morph:
        return False

    word_lower = word.lower().replace('ё', 'е')
    parsed = morph.parse(word_lower)

    for p in parsed:
        tag = p.tag
        # Проверяем теги имён собственных
        if 'Name' in tag or 'Surn' in tag or 'Patr' in tag or 'Geox' in tag:
            return True

    return False


def is_capitalized_not_start(word: str, position: int, text: str) -> bool:
    """
    Проверяет, является ли слово с заглавной буквы не в начале предложения.

    Args:
        word: слово для проверки
        position: позиция слова в тексте
        text: полный текст

    Returns:
        True если слово с заглавной буквы, но не в начале предложения
    """
    if not word or not word[0].isupper():
        return False

    # Проверяем, что это не начало текста
    if position == 0:
        return False

    # Проверяем символы перед словом
    before = text[:position].rstrip()
    if not before:
        return False

    last_char = before[-1]

    # Если перед словом точка/!/? — это начало предложения
    if last_char in '.!?':
        return False

    # Если перед словом открывающая кавычка после точки
    if last_char in '«"\'':
        before_quote = before[:-1].rstrip()
        if before_quote and before_quote[-1] in '.!?':
            return False

    return True


def extract_capitalized_words(text: str) -> Set[str]:
    """
    Извлекает слова с заглавной буквы, которые не в начале предложения.

    Returns:
        Множество потенциальных имён собственных
    """
    names = set()

    # Паттерн для слов с заглавной буквы
    pattern = r'\b([А-ЯЁ][а-яё]+)\b'

    for match in re.finditer(pattern, text):
        word = match.group(1)
        position = match.start()

        # Проверяем, что это не начало предложения
        if is_capitalized_not_start(word, position, text):
            word_lower = word.lower().replace('ё', 'е')

            # Исключаем стоп-слова
            if word_lower not in STOP_WORDS:
                names.add(word)

    return names


def extract_names_by_morph(text: str) -> Set[str]:
    """
    Извлекает имена через морфологический анализ.

    Находит все слова и проверяет их теги в pymorphy2.
    """
    if not HAS_PYMORPHY:
        return set()

    names = set()

    # Находим все слова
    words = re.findall(r'\b([А-ЯЁа-яё]+)\b', text)

    for word in words:
        if len(word) >= 2 and is_proper_name_by_morph(word):
            word_lower = word.lower().replace('ё', 'е')
            if word_lower not in STOP_WORDS:
                names.add(word.capitalize())

    return names


def extract_name_patterns(text: str) -> Set[str]:
    """
    Извлекает имена по паттернам (Имя Отчество, Имя Фамилия).
    """
    names = set()

    # Паттерн: Имя Отчество (Иван Петрович)
    pattern_patronymic = r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:ович|евич|ич|овна|евна|ична|инична))\b'
    for match in re.finditer(pattern_patronymic, text):
        name, patronymic = match.groups()
        names.add(name)
        names.add(patronymic)

    # Паттерн: Два слова с заглавной подряд (Иван Иванов)
    pattern_two_caps = r'\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b'
    for match in re.finditer(pattern_two_caps, text):
        word1, word2 = match.groups()
        word1_lower = word1.lower().replace('ё', 'е')
        word2_lower = word2.lower().replace('ё', 'е')

        # Исключаем стоп-слова
        if word1_lower not in STOP_WORDS:
            names.add(word1)
        if word2_lower not in STOP_WORDS:
            names.add(word2)

    return names


def extract_names_from_text(text: str, use_morph: bool = True) -> Set[str]:
    """
    Извлекает все потенциальные имена собственные из текста.

    Args:
        text: текст для анализа
        use_morph: использовать ли морфологический анализ

    Returns:
        Множество имён собственных
    """
    names = set()

    # 1. Слова с заглавной буквы не в начале предложения
    names |= extract_capitalized_words(text)

    # 2. Морфологический анализ
    if use_morph and HAS_PYMORPHY:
        names |= extract_names_by_morph(text)

    # 3. Паттерны имён
    names |= extract_name_patterns(text)

    # Фильтруем слишком короткие
    names = {n for n in names if len(n) >= 2}

    return names


def extract_names_from_docx(docx_path: str, use_morph: bool = True) -> Set[str]:
    """
    Извлекает имена собственные из DOCX файла.

    Args:
        docx_path: путь к DOCX файлу
        use_morph: использовать ли морфологический анализ

    Returns:
        Множество имён собственных
    """
    if not HAS_DOCX:
        raise ImportError("python-docx не установлен. Установите: pip install python-docx")

    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {docx_path}")

    # Читаем документ
    doc = Document(str(path))

    # Собираем весь текст
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = '\n'.join(paragraphs)

    return extract_names_from_text(full_text, use_morph=use_morph)


def extract_names_from_txt(txt_path: str, use_morph: bool = True) -> Set[str]:
    """
    Извлекает имена собственные из TXT файла.
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {txt_path}")

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    return extract_names_from_text(text, use_morph=use_morph)


def extract_names_from_file(file_path: str, use_morph: bool = True) -> Set[str]:
    """
    Извлекает имена собственные из файла (DOCX или TXT).
    """
    path = Path(file_path)

    if path.suffix.lower() == '.docx':
        return extract_names_from_docx(file_path, use_morph=use_morph)
    else:
        return extract_names_from_txt(file_path, use_morph=use_morph)


def save_names_to_protected(names: Set[str], protected_path: str = None,
                            append: bool = True, force: bool = False):
    """
    Сохраняет имена в файл защищённых слов.

    Args:
        names: множество имён
        protected_path: путь к файлу защищённых слов (по умолчанию NAMES_DICT)
        append: добавить к существующим (True) или перезаписать (False)
        force: перезаписать без предупреждения

    Returns:
        количество добавленных имён
    """
    # Используем NAMES_DICT по умолчанию
    if protected_path is None:
        path = NAMES_DICT
    else:
        path = Path(protected_path)

    existing = set()
    if append and path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = {line.strip().lower() for line in f if line.strip() and not line.startswith('#')}

    # Добавляем только новые имена
    new_names = {n for n in names if n.lower() not in existing}

    if not new_names:
        print(f"  Нет новых имён для добавления")
        return 0

    # Проверяем существование файла при перезаписи
    if not append and path.exists() and not force:
        if not check_file_exists(path, action='ask'):
            return 0

    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        if append and path.exists():
            f.write('\n')
        f.write('# Автоизвлечённые имена\n')
        for name in sorted(new_names):
            f.write(f'{name}\n')

    print(f"  Добавлено имён: {len(new_names)}")
    return len(new_names)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI для извлечения имён"""
    import argparse

    # Формируем help с путём к NAMES_DICT
    names_dict_path = str(NAMES_DICT) if HAS_CONFIG else "Словари/Словарь_имён_персонажей.txt"

    parser = argparse.ArgumentParser(
        description='Извлечение имён собственных из текста',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Примеры:
  python proper_names_extractor.py глава.docx
  python proper_names_extractor.py глава.docx --to-names-dict
  python proper_names_extractor.py глава.docx --output имена.txt --force

Путь к словарю имён по умолчанию:
  {names_dict_path}
        """
    )
    parser.add_argument('input', help='Входной файл (DOCX или TXT)')
    parser.add_argument('--output', '-o', help='Файл для сохранения имён')
    parser.add_argument('--append', '-a', action='store_true',
                        help='Добавить к существующим (по умолчанию: перезаписать)')
    parser.add_argument('--no-morph', action='store_true',
                        help='Не использовать морфологический анализ')
    parser.add_argument('--protected', '-p',
                        help='Путь к файлу защищённых слов (добавить туда)')
    parser.add_argument('--to-names-dict', action='store_true',
                        help=f'Добавить имена в словарь имён персонажей ({NAMES_DICT.name})')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Перезаписать существующие файлы')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ИЗВЛЕЧЕНИЕ ИМЁН СОБСТВЕННЫХ v{VERSION}")
    print(f"{'='*60}")
    print(f"  Файл: {args.input}")
    print(f"  Морфология: {'да' if not args.no_morph else 'нет'}")
    if HAS_PYMORPHY:
        print(f"  pymorphy2: доступен")
    else:
        print(f"  pymorphy2: недоступен (установите: pip install pymorphy2)")

    try:
        names = extract_names_from_file(
            args.input,
            use_morph=not args.no_morph
        )

        print(f"\n  Найдено имён: {len(names)}")

        if names:
            print(f"\n  Имена:")
            for name in sorted(names)[:50]:
                print(f"    - {name}")
            if len(names) > 50:
                print(f"    ... и ещё {len(names) - 50}")

        # Сохраняем в файл
        if args.output:
            output_path = Path(args.output)

            # Проверяем существование
            if output_path.exists() and not args.force:
                check_file_exists(output_path, action='ask')

            with open(output_path, 'w', encoding='utf-8') as f:
                for name in sorted(names):
                    f.write(f'{name}\n')
            print(f"\n  Сохранено в: {output_path}")

        # Добавляем в защищённые слова (--protected)
        if args.protected:
            save_names_to_protected(names, args.protected, append=True, force=args.force)
            print(f"  Обновлён файл: {args.protected}")

        # Добавляем в словарь имён персонажей (--to-names-dict)
        if args.to_names_dict:
            save_names_to_protected(names, str(NAMES_DICT), append=True, force=args.force)
            print(f"  Обновлён словарь: {NAMES_DICT}")

        print(f"\n{'='*60}")
        print("✓ Готово!")

    except Exception as e:
        print(f"\n  ✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
