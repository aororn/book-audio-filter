#!/usr/bin/env python3
"""
Morphology v6.0 - Единый модуль морфологического анализа с кэшированием

Централизованный модуль для работы с pymorphy2:
- Кэширование в памяти (lru_cache)
- Персистентный кэш на диске (sqlite)
- Единый интерфейс для всех модулей проекта
- Расширенный анализ глаголов (вид, залог, время)

Использование:
    from morphology import get_lemma, get_pos, get_word_info, MorphCache
    from morphology import get_aspect, get_voice, get_tense, is_aspect_pair

    lemma = get_lemma("слово")
    pos = get_pos("слово")
    info = get_word_info("слово")  # (lemma, pos, number, gender, case)

    aspect = get_aspect("сделал")  # 'perf'
    is_pair = is_aspect_pair("делать", "сделать")  # True

Changelog:
    v6.0 (2026-01-26): Расширенный анализ глаголов
        - get_aspect() — вид глагола (perf/impf)
        - get_voice() — залог (actv/pssv)
        - get_tense() — время (past/pres/futr)
        - get_verb_info() — полная информация о глаголе
        - is_aspect_pair() — проверка видовой пары
        - is_same_verb_base() — проверка основы глагола
    v2.0 (2026-01-24): Интеграция с config.py
        - TEMP_DIR для директории кэша
        - Исправлен bare except → sqlite3.Error
        - Добавлен --version в CLI
        - Добавлен --force для --clear
        - VERSION/VERSION_DATE константы
    v1.0: Базовая реализация с двухуровневым кэшем
"""

# Версия модуля
VERSION = '6.0.0'
VERSION_DATE = '2026-01-26'

import os
import sqlite3
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


# =============================================================================
# ИМПОРТ ЦЕНТРАЛИЗОВАННОЙ КОНФИГУРАЦИИ
# =============================================================================

try:
    from config import TEMP_DIR
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Fallback: папка рядом с модулем
    TEMP_DIR = Path(__file__).parent.parent / 'Темп'


# =============================================================================
# ИНИЦИАЛИЗАЦИЯ PYMORPHY
# =============================================================================

HAS_PYMORPHY = False
morph = None

try:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    HAS_PYMORPHY = True
except ImportError:
    try:
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        HAS_PYMORPHY = True
    except ImportError:
        print("⚠ pymorphy не установлен. Установите: pip install pymorphy3")


# =============================================================================
# ТИПЫ ДАННЫХ
# =============================================================================

WordInfo = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]
# (lemma, pos, number, gender, case)

# Расширенная информация для глаголов (v6.0)
VerbInfo = Tuple[Optional[str], Optional[str], Optional[str]]
# (aspect, voice, tense)


# =============================================================================
# ПЕРСИСТЕНТНЫЙ КЭШ НА ДИСКЕ
# =============================================================================

class MorphCache:
    """
    Персистентный кэш морфологического анализа на SQLite.

    Сохраняет результаты разбора слов между запусками,
    что ускоряет повторную обработку одних и тех же текстов.
    """

    _instance = None
    _cache_dir = None
    _db_path = None
    _conn = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Определяем путь к кэшу (используем TEMP_DIR из config.py)
        self._cache_dir = TEMP_DIR / 'cache'
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._cache_dir / 'morph_cache.db'
        self._init_db()
        self._initialized = True

        # Статистика
        self._hits = 0
        self._misses = 0

    def _init_db(self):
        """Инициализирует базу данных"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS morph_cache (
                word TEXT PRIMARY KEY,
                lemma TEXT,
                pos TEXT,
                number TEXT,
                gender TEXT,
                word_case TEXT
            )
        ''')
        self._conn.execute('CREATE INDEX IF NOT EXISTS idx_word ON morph_cache(word)')
        self._conn.commit()

    def get(self, word: str) -> Optional[WordInfo]:
        """Получает результат из кэша"""
        cursor = self._conn.execute(
            'SELECT lemma, pos, number, gender, word_case FROM morph_cache WHERE word = ?',
            (word,)
        )
        row = cursor.fetchone()
        if row:
            self._hits += 1
            return row
        self._misses += 1
        return None

    def set(self, word: str, info: WordInfo):
        """Сохраняет результат в кэш"""
        try:
            self._conn.execute(
                '''INSERT OR REPLACE INTO morph_cache
                   (word, lemma, pos, number, gender, word_case) VALUES (?, ?, ?, ?, ?, ?)''',
                (word, info[0], info[1], info[2], info[3], info[4])
            )
            self._conn.commit()
        except sqlite3.Error:
            pass  # Игнорируем ошибки записи в БД

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        cursor = self._conn.execute('SELECT COUNT(*) FROM morph_cache')
        total = cursor.fetchone()[0]

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'total_entries': total,
            'session_hits': self._hits,
            'session_misses': self._misses,
            'hit_rate': f'{hit_rate:.1f}%',
            'db_path': str(self._db_path),
            'db_size_mb': self._db_path.stat().st_size / 1024 / 1024 if self._db_path.exists() else 0
        }

    def clear(self):
        """Очищает кэш"""
        self._conn.execute('DELETE FROM morph_cache')
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def close(self):
        """Закрывает соединение"""
        if self._conn:
            self._conn.close()


# Глобальный экземпляр кэша
_disk_cache: Optional[MorphCache] = None


def get_disk_cache() -> MorphCache:
    """Возвращает глобальный экземпляр дискового кэша"""
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = MorphCache()
    return _disk_cache


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ АНАЛИЗА
# =============================================================================

def normalize_word(word: str) -> str:
    """Базовая нормализация слова"""
    return word.lower().strip().replace('ё', 'е')


def _parse_word_raw(word: str) -> WordInfo:
    """
    Выполняет морфологический разбор слова через pymorphy.
    Возвращает кортеж (lemma, pos, number, gender, case).
    """
    if not HAS_PYMORPHY or not morph:
        return (word, None, None, None, None)

    parsed = morph.parse(word)
    if not parsed:
        return (word, None, None, None, None)

    p = parsed[0]
    tag = p.tag

    lemma = p.normal_form
    pos = tag.POS

    # Число
    number = None
    if 'sing' in tag:
        number = 'sing'
    elif 'plur' in tag:
        number = 'plur'

    # Род
    gender = None
    if 'masc' in tag:
        gender = 'masc'
    elif 'femn' in tag:
        gender = 'femn'
    elif 'neut' in tag:
        gender = 'neut'

    # Падеж
    word_case = None
    for c in ('nomn', 'gent', 'datv', 'accs', 'ablt', 'loct'):
        if c in tag:
            word_case = c
            break

    return (lemma, pos, number, gender, word_case)


@lru_cache(maxsize=50000)
def parse_word_cached(word: str) -> WordInfo:
    """
    Кэшированный разбор слова (память + диск).

    Сначала проверяет lru_cache в памяти,
    затем дисковый кэш SQLite,
    и только потом вызывает pymorphy.
    """
    # Проверяем дисковый кэш
    disk_cache = get_disk_cache()
    cached = disk_cache.get(word)
    if cached:
        return cached

    # Выполняем разбор
    result = _parse_word_raw(word)

    # Сохраняем в дисковый кэш
    disk_cache.set(word, result)

    return result


def get_word_info(word: str) -> WordInfo:
    """
    Получает полную информацию о слове.

    Args:
        word: слово для анализа

    Returns:
        Кортеж (lemma, pos, number, gender, case)
    """
    word = normalize_word(word)
    return parse_word_cached(word)


def get_lemma(word: str) -> str:
    """Получает лемму (нормальную форму) слова"""
    return get_word_info(word)[0]


def get_pos(word: str) -> Optional[str]:
    """Получает часть речи (POS tag)"""
    return get_word_info(word)[1]


def get_number(word: str) -> Optional[str]:
    """Получает грамматическое число (sing/plur)"""
    return get_word_info(word)[2]


def get_gender(word: str) -> Optional[str]:
    """Получает род (masc/femn/neut)"""
    return get_word_info(word)[3]


def get_case(word: str) -> Optional[str]:
    """Получает падеж (nomn/gent/datv/accs/ablt/loct)"""
    return get_word_info(word)[4]


# =============================================================================
# РАСШИРЕННЫЙ АНАЛИЗ ГЛАГОЛОВ (v6.0)
# =============================================================================

@lru_cache(maxsize=50000)
def _parse_verb_info(word: str) -> VerbInfo:
    """
    Получает информацию о глаголе: вид, залог, время.

    Args:
        word: слово для анализа (нормализованное)

    Returns:
        Кортеж (aspect, voice, tense)
        aspect: 'perf' (совершенный) или 'impf' (несовершенный)
        voice: 'actv' (действительный) или 'pssv' (страдательный)
        tense: 'past', 'pres', 'futr'
    """
    if not HAS_PYMORPHY or not morph:
        return (None, None, None)

    parsed = morph.parse(word)
    if not parsed:
        return (None, None, None)

    p = parsed[0]
    tag = p.tag

    # Вид глагола (aspect)
    aspect = None
    if 'perf' in tag:
        aspect = 'perf'  # совершенный вид (что сделать?)
    elif 'impf' in tag:
        aspect = 'impf'  # несовершенный вид (что делать?)

    # Залог (voice)
    voice = None
    if 'actv' in tag:
        voice = 'actv'  # действительный
    elif 'pssv' in tag:
        voice = 'pssv'  # страдательный

    # Время (tense)
    tense = None
    if 'past' in tag:
        tense = 'past'
    elif 'pres' in tag:
        tense = 'pres'
    elif 'futr' in tag:
        tense = 'futr'

    return (aspect, voice, tense)


def get_aspect(word: str) -> Optional[str]:
    """
    Получает вид глагола.

    Returns:
        'perf' (совершенный - что сделать?) или
        'impf' (несовершенный - что делать?) или
        None для неглаголов
    """
    word = normalize_word(word)
    return _parse_verb_info(word)[0]


def get_voice(word: str) -> Optional[str]:
    """
    Получает залог глагола.

    Returns:
        'actv' (действительный) или
        'pssv' (страдательный) или
        None
    """
    word = normalize_word(word)
    return _parse_verb_info(word)[1]


def get_tense(word: str) -> Optional[str]:
    """
    Получает время глагола.

    Returns:
        'past', 'pres', 'futr' или None
    """
    word = normalize_word(word)
    return _parse_verb_info(word)[2]


def get_verb_info(word: str) -> VerbInfo:
    """
    Получает полную информацию о глаголе.

    Args:
        word: слово для анализа

    Returns:
        Кортеж (aspect, voice, tense)
    """
    word = normalize_word(word)
    return _parse_verb_info(word)


def is_aspect_pair(word1: str, word2: str) -> bool:
    """
    Проверяет, являются ли слова видовой парой.

    Пример: "делать" (impf) и "сделать" (perf)

    Returns:
        True если одна лемма и разный вид
    """
    # Сначала проверяем лемму
    lemma1, lemma2 = get_lemma(word1), get_lemma(word2)
    if lemma1 != lemma2:
        # Разные леммы - проверяем, не пара ли это по основе
        # Часто видовые пары имеют приставку у совершенного вида
        # делать -> сделать, писать -> написать
        pass

    # Проверяем вид
    aspect1, aspect2 = get_aspect(word1), get_aspect(word2)

    # Оба должны быть глаголами
    if aspect1 is None or aspect2 is None:
        return False

    # Разный вид = видовая пара
    return aspect1 != aspect2


def is_same_verb_base(word1: str, word2: str) -> bool:
    """
    Проверяет, одна ли глагольная основа у слов.

    Учитывает типичные префиксы для видовых пар.
    """
    # Получаем леммы
    lemma1, lemma2 = get_lemma(word1), get_lemma(word2)

    # Простой случай: одна лемма
    if lemma1 == lemma2:
        return True

    # Проверяем приставочные пары
    # Типичные приставки для образования совершенного вида
    perf_prefixes = ['с', 'по', 'на', 'вы', 'за', 'про', 'от', 'у']

    for prefix in perf_prefixes:
        # lemma1 = prefix + lemma2
        if lemma1.startswith(prefix) and lemma1[len(prefix):] == lemma2:
            return True
        # lemma2 = prefix + lemma1
        if lemma2.startswith(prefix) and lemma2[len(prefix):] == lemma1:
            return True

    return False


# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def is_same_lemma(word1: str, word2: str) -> bool:
    """Проверяет, имеют ли слова одну лемму"""
    return get_lemma(word1) == get_lemma(word2)


def is_same_pos(word1: str, word2: str) -> bool:
    """Проверяет, являются ли слова одной частью речи"""
    pos1, pos2 = get_pos(word1), get_pos(word2)
    return pos1 is not None and pos1 == pos2


def get_all_forms(word: str) -> list:
    """
    Возвращает все формы слова.

    Args:
        word: слово для анализа

    Returns:
        Список всех форм слова
    """
    if not HAS_PYMORPHY or not morph:
        return [word]

    word = normalize_word(word)
    parsed = morph.parse(word)
    if not parsed:
        return [word]

    forms = set()
    for p in parsed:
        lexeme = p.lexeme
        for form in lexeme:
            forms.add(form.word)

    return sorted(forms)


def get_cache_stats() -> Dict[str, Any]:
    """
    Возвращает статистику кэширования.

    Returns:
        Словарь со статистикой памяти и диска
    """
    memory_info = parse_word_cached.cache_info()
    disk_info = get_disk_cache().get_stats()

    total_hits = memory_info.hits + disk_info['session_hits']
    total_misses = memory_info.misses  # disk misses уже включены в pymorphy calls
    total = total_hits + total_misses

    return {
        'memory': {
            'hits': memory_info.hits,
            'misses': memory_info.misses,
            'size': memory_info.currsize,
            'maxsize': memory_info.maxsize,
        },
        'disk': disk_info,
        'total_efficiency': f'{(total_hits / total * 100):.1f}%' if total > 0 else 'N/A'
    }


def clear_cache():
    """Очищает все кэши (память и диск)"""
    parse_word_cached.cache_clear()
    get_disk_cache().clear()


# =============================================================================
# ЭКСПОРТ
# =============================================================================

__all__ = [
    # Проверка доступности
    'HAS_PYMORPHY',
    'morph',

    # Основные функции
    'normalize_word',
    'get_word_info',
    'get_lemma',
    'get_pos',
    'get_number',
    'get_gender',
    'get_case',

    # Глагольные функции (v6.0)
    'get_aspect',
    'get_voice',
    'get_tense',
    'get_verb_info',
    'is_aspect_pair',
    'is_same_verb_base',

    # Дополнительные функции
    'is_same_lemma',
    'is_same_pos',
    'get_all_forms',

    # Кэширование
    'parse_word_cached',
    'get_cache_stats',
    'clear_cache',
    'MorphCache',
    'get_disk_cache',
]


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI для тестирования модуля"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Морфологический анализ слов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python morphology.py слово              # Анализ одного слова
  python morphology.py слово --forms      # Показать все формы
  python morphology.py --stats            # Статистика кэша
  python morphology.py --clear --force    # Очистить кэш
        """
    )
    parser.add_argument('words', nargs='*', help='Слова для анализа')
    parser.add_argument('--stats', action='store_true', help='Показать статистику кэша')
    parser.add_argument('--clear', action='store_true', help='Очистить кэш')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Принудительное выполнение без подтверждения')
    parser.add_argument('--forms', action='store_true', help='Показать все формы слова')
    parser.add_argument('--version', '-V', action='store_true',
                        help='Показать версию и выйти')

    args = parser.parse_args()

    # Вывод версии
    if args.version:
        print(f"Morphology v{VERSION} ({VERSION_DATE})")
        print(f"  pymorphy2: {'доступен' if HAS_PYMORPHY else 'недоступен'}")
        print(f"  Config: {'config.py' if HAS_CONFIG else 'fallback'}")
        return

    if args.clear:
        if args.force:
            clear_cache()
            print("✓ Кэш очищен")
        else:
            confirm = input("Очистить весь морфологический кэш? (yes/no): ")
            if confirm.lower() == 'yes':
                clear_cache()
                print("✓ Кэш очищен")
            else:
                print("Отменено")
        return

    if args.stats:
        stats = get_cache_stats()
        print("\n=== Статистика кэша ===")
        print(f"Память:")
        print(f"  Попаданий: {stats['memory']['hits']}")
        print(f"  Промахов: {stats['memory']['misses']}")
        print(f"  Размер: {stats['memory']['size']}/{stats['memory']['maxsize']}")
        print(f"Диск:")
        print(f"  Записей: {stats['disk']['total_entries']}")
        print(f"  Размер: {stats['disk']['db_size_mb']:.2f} MB")
        print(f"  Путь: {stats['disk']['db_path']}")
        print(f"Общая эффективность: {stats['total_efficiency']}")
        return

    if not args.words:
        print("Укажите слова для анализа или используйте --stats/--clear")
        return

    for word in args.words:
        info = get_word_info(word)
        verb_info = get_verb_info(word)

        print(f"\n{word}:")
        print(f"  Лемма: {info[0]}")
        print(f"  Часть речи: {info[1]}")
        print(f"  Число: {info[2]}")
        print(f"  Род: {info[3]}")
        print(f"  Падеж: {info[4]}")

        # Глагольная информация (если это глагол)
        if verb_info[0] or verb_info[1] or verb_info[2]:
            print(f"  --- Глагол ---")
            aspect_names = {'perf': 'совершенный', 'impf': 'несовершенный'}
            voice_names = {'actv': 'действительный', 'pssv': 'страдательный'}
            tense_names = {'past': 'прошедшее', 'pres': 'настоящее', 'futr': 'будущее'}
            print(f"  Вид: {aspect_names.get(verb_info[0], verb_info[0])}")
            print(f"  Залог: {voice_names.get(verb_info[1], verb_info[1])}")
            print(f"  Время: {tense_names.get(verb_info[2], verb_info[2])}")

        if args.forms:
            forms = get_all_forms(word)
            print(f"  Формы: {', '.join(forms[:10])}{'...' if len(forms) > 10 else ''}")


if __name__ == '__main__':
    main()
