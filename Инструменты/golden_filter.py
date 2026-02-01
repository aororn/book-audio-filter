#!/usr/bin/env python3
"""
Golden Filter v5.0 — Обёртка обратной совместимости.

Начиная с v5.0, фильтрация разделена на модульный пакет filters/:
    filters/constants.py   — словари и константы
    filters/comparison.py  — функции сравнения слов
    filters/detectors.py   — специализированные детекторы
    filters/engine.py      — движок фильтрации (should_filter_error, filter_errors, filter_report)

Этот файл реэкспортирует всё из пакета filters для обратной совместимости.
Весь существующий код, импортирующий из golden_filter, продолжит работать.

Использование:
    python golden_filter.py отчет.json --output отфильтрованный.json
"""

import argparse
import json
import os
from pathlib import Path

# Реэкспорт всего из пакета filters
from filters import (
    # Движок
    should_filter_error, filter_errors, filter_report,
    # Сравнение
    normalize_word, levenshtein_distance, levenshtein_ratio,
    is_homophone_match, is_grammar_ending_match, is_case_form_match,
    is_adverb_adjective_match, is_verb_gerund_safe_match,
    is_short_full_adjective_match, is_lemma_match,
    is_similar_by_levenshtein, is_yandex_typical_error,
    is_prefix_variant, is_interjection,
    get_word_info, get_lemma, get_pos, get_number, get_gender,
    parse_word_cached,
    HAS_PYMORPHY, HAS_RAPIDFUZZ,
    # Детекторы
    is_yandex_name_error, is_merged_word_error, is_compound_word_match,
    is_split_name_insertion, is_compound_prefix_insertion,
    is_split_compound_insertion, is_context_artifact,
    detect_alignment_chains,
    load_character_names_dictionary, load_base_character_names,
    FULL_CHARACTER_NAMES, CHARACTER_NAMES_BASE,
    # Константы
    HOMOPHONES, GRAMMAR_ENDINGS, WEAK_WORDS, PROTECTED_WORDS,
    INTERJECTIONS, YANDEX_TYPICAL_ERRORS, YANDEX_NAME_ERRORS,
    YANDEX_PREFIX_ERRORS, CHARACTER_NAMES,
)

# Обратная совместимость: загрузка словарей
try:
    from config import (
        FileNaming, GoldenFilterConfig,
        NAMES_DICT, PROTECTED_WORDS as PROTECTED_WORDS_PATH,
        READER_ERRORS as READER_ERRORS_PATH,
        DICTIONARIES_DIR, check_file_exists
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    PROTECTED_WORDS_PATH = None
    READER_ERRORS_PATH = None


def load_reader_errors(path: str) -> None:
    """Загружает типичные ошибки чтеца."""
    from filters.constants import YANDEX_TYPICAL_ERRORS as errors_set

    if not os.path.exists(path):
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if isinstance(item, list) and len(item) == 2:
            errors_set.add((item[0].lower(), item[1].lower()))
        elif isinstance(item, dict):
            wrong = item.get('wrong', item.get('heard', ''))
            correct = item.get('correct', item.get('actual', ''))
            if wrong and correct:
                errors_set.add((wrong.lower(), correct.lower()))

    print(f"  Загружено ошибок чтеца: {len(errors_set)}")


def main():
    parser = argparse.ArgumentParser(
        description='Golden Filter v5.0 — модульная фильтрация ошибок транскрипции',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Версия 5.0 — модульная архитектура:
  - Пакет filters/ с разделением на constants, comparison, detectors, engine
  - Обратная совместимость: все импорты из golden_filter продолжают работать
  - Интеграция с config.py (FileNaming, GoldenFilterConfig)

Примеры:
  python golden_filter.py 01_compared.json
  python golden_filter.py 01_compared.json --force
  python golden_filter.py 01_compared.json --levenshtein 1
        """
    )
    parser.add_argument('report', help='JSON файл с отчётом ошибок (_compared.json)')
    parser.add_argument('--output', '-o', help='Выходной файл (по умолчанию {chapter}_filtered.json)')
    parser.add_argument('--config', '-c', help='JSON конфигурация')
    parser.add_argument('--levenshtein', '-l', type=int, default=None,
                        help='Порог Левенштейна (по умолчанию из config.py или 2)')
    parser.add_argument('--no-lemma', action='store_true', help='Отключить лемматизацию')
    parser.add_argument('--no-homophones', action='store_true', help='Отключить омофоны')
    parser.add_argument('--reader-errors', '-r', help='JSON с ошибками чтеца')
    parser.add_argument('--protected', '-p', help='Файл с защищёнными словами')
    parser.add_argument('--force', action='store_true', help='Перезаписать существующие файлы')

    args = parser.parse_args()

    # Автозагрузка словарей
    if args.reader_errors:
        load_reader_errors(args.reader_errors)
    elif HAS_CONFIG and READER_ERRORS_PATH and READER_ERRORS_PATH.exists():
        load_reader_errors(str(READER_ERRORS_PATH))

    if args.protected:
        load_protected_words(args.protected)
    elif HAS_CONFIG and PROTECTED_WORDS_PATH and PROTECTED_WORDS_PATH.exists():
        load_protected_words(str(PROTECTED_WORDS_PATH))

    if HAS_CONFIG:
        default_threshold = GoldenFilterConfig.LEVENSHTEIN_THRESHOLD
        default_lemma = GoldenFilterConfig.USE_LEMMATIZATION
        default_homophones = GoldenFilterConfig.USE_HOMOPHONES
    else:
        default_threshold = 2
        default_lemma = True
        default_homophones = True

    config = {
        'levenshtein_threshold': args.levenshtein if args.levenshtein is not None else default_threshold,
        'use_lemmatization': (not args.no_lemma) and default_lemma,
        'use_homophones': (not args.no_homophones) and default_homophones,
    }

    try:
        filter_report(args.report, output_path=args.output, config_path=args.config,
                      force=args.force, **config)
        print("✓ Готово!")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
