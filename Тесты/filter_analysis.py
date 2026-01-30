#!/usr/bin/env python3
"""
Фреймворк анализа эффективности фильтров v1.0

Анализирует работу каждого уровня фильтрации:
- Прогоняет транскрипции через полный пайплайн
- Логирует каждый уровень фильтрации отдельно
- Считает: сколько ошибок отфильтровал каждый уровень
- Проверяет: не зацепил ли уровень golden ошибки
- Строит матрицу эффективности

Использование:
    python Тесты/filter_analysis.py                    # анализ всех транскрипций
    python Тесты/filter_analysis.py --chapter 1       # только глава 1
    python Тесты/filter_analysis.py --transcript PATH  # конкретная транскрипция
    python Тесты/filter_analysis.py --summary          # сводка по всем прогонам
    python Тесты/filter_analysis.py --matrix           # матрица эффективности

Версия: 1.0.0 (2026-01-30)
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-30'

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

# Добавляем путь к модулям
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'Инструменты'))

from config import (
    RESULTS_DIR, TESTS_DIR, TRANSCRIPTIONS_DIR,
    CHAPTERS_DIR, AUDIO_DIR, FileNaming
)


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

ANALYSIS_DIR = TESTS_DIR / 'Анализ_фильтров'
GOLDEN_FILES = {
    '1': TESTS_DIR / 'золотой_стандарт_глава1.json',
    '2': TESTS_DIR / 'золотой_стандарт_глава2.json',
    '3': TESTS_DIR / 'золотой_стандарт_глава3.json',
    '4': TESTS_DIR / 'золотой_стандарт_глава4.json',
}

# Порядок уровней фильтрации (из engine.py)
FILTER_LEVELS = [
    # Защитные уровни (возвращают False = НЕ фильтровать)
    ('PROTECTED_hard_negative', -1, 'protection', 'Известные пары путаницы'),
    ('PROTECTED_semantic_slip', -0.5, 'protection', 'Семантические оговорки'),

    # Морфология
    ('morpho_same_form', 0, 'morpho', 'Идентичные формы'),
    ('morpho_proper_name', 0, 'morpho', 'Имена собственные'),

    # Ранние правила
    ('safe_ending_transition', 0.3, 'alignment', 'Безопасные окончания'),
    ('yandex_phonetic_pair', 0.5, 'phonetic', 'Фонетические пары Яндекса'),
    ('alignment_artifact', 0.6, 'alignment', 'Артефакты выравнивания (подстрока)'),
    ('alignment_artifact_substring', 0.6, 'alignment', 'Артефакты (подстрока длинного)'),

    # Этап 0: Артефакты алгоритма
    ('alignment_start_artifact', 1, 'alignment', 'Удаление в начале (t=0)'),
    ('character_name_unrecognized', 1, 'names', 'Имена персонажей не распознаны'),
    ('split_name_insertion', 1, 'split', 'Разбитое имя'),
    ('split_name', 1, 'split', 'Разбитое имя (детектор)'),
    ('split_compound', 1, 'split', 'Разбитое составное слово'),
    ('split_word_yandex', 1, 'split', 'Яндекс разбил слово'),
    ('split_suffix_insertion', 1, 'split', 'Суффикс как вставка'),
    ('split_word_fragment', 1, 'split', 'Фрагмент разбитого слова'),
    ('interrogative_split_to', 1, 'split', 'Разбитое дефисное (кто-то→кто то)'),
    ('compound_particle_to', 1, 'split', 'Частица "то"'),

    # Этап 3: Междометия и артефакты
    ('interjection', 3, 'weak', 'Междометия'),
    ('single_consonant_artifact', 3, 'alignment', 'Однобуквенные согласные'),
    ('misrecognition_artifact', 3, 'alignment', 'Похоже на соседнее слово'),
    ('unknown_word_artifact', 3, 'alignment', 'Неизвестное слово (UNKN)'),
    ('rare_adverb', 3, 'weak', 'Редкие наречия'),
    ('sentence_start_weak', 3, 'weak', 'Слабое слово в начале предложения'),
    ('hyphenated_part', 3, 'split', 'Часть дефисного слова'),
    ('compound_word_part', 3, 'split', 'Часть составного слова'),

    # Этап 4: Контекстные
    ('context_artifact', 4, 'context', 'Артефакт контекста'),

    # Уровень 1: Защищённые слова
    ('yandex_typical', 5, 'yandex', 'Типичные ошибки Яндекса'),
    ('same_lemma', 5, 'morpho', 'Одинаковая лемма'),
    ('yandex_name_error', 5, 'names', 'Ошибка в имени'),
    ('levenshtein_protected', 5, 'phonetic', 'Левенштейн ≤1 (защищённые)'),

    # Уровень 2: Слабые слова
    ('alignment_artifact', 6, 'weak', 'Слабые DEL'),
    ('sentence_start_conjunction', 6, 'weak', 'Союз в начале предложения'),
    ('split_word_insertion', 6, 'split', 'Вставка части слова'),
    ('yandex_merge_artifact', 6, 'yandex', 'Яндекс слил слова'),
    ('yandex_truncate_artifact', 6, 'yandex', 'Яндекс обрезал'),
    ('yandex_expand_artifact', 6, 'yandex', 'Яндекс расширил'),
    ('yandex_i_ya_confusion', 6, 'phonetic', 'Путаница и↔я (контекст)'),
    ('yandex_i_ya_verb_context', 6, 'phonetic', 'Путаница и↔я (глаголы)'),
    ('weak_words_identical', 6, 'weak', 'Слабые одинаковые'),
    ('weak_words_same_lemma', 6, 'weak', 'Слабые одной леммы'),

    # Уровень 3: Substitution
    ('identical_normalized', 7, 'normalization', 'Идентичные после нормализации'),
    ('homophone', 7, 'phonetic', 'Омофоны'),
    ('compound_word', 7, 'split', 'Составное слово'),
    ('merged_word', 7, 'split', 'Слияние слов'),
    ('case_form', 7, 'morpho', 'Падежная форма'),
    ('adverb_adjective', 7, 'morpho', 'Наречие↔прилагательное'),
    ('short_full_adjective', 7, 'morpho', 'Краткое↔полное прилагательное'),
    ('verb_gerund_safe', 7, 'morpho', 'Глагол↔деепричастие'),

    # Цепочки
    ('alignment_chain', 8, 'chain', 'Цепочка выравнивания'),
    ('linked_prefix_error', 8, 'chain', 'Связанная ошибка приставки'),

    # ML
    ('ml_classifier', 10, 'ml', 'ML-классификатор'),

    # SmartFilter (отключён)
    ('smart_filter', 11, 'smart', 'SmartFilter (скоринг)'),
]

# Группы фильтров для сводки
FILTER_GROUPS = {
    'protection': 'Защитные слои',
    'morpho': 'Морфология',
    'alignment': 'Артефакты выравнивания',
    'phonetic': 'Фонетика',
    'split': 'Разбитые слова',
    'weak': 'Слабые слова',
    'names': 'Имена персонажей',
    'yandex': 'Типичные ошибки Яндекса',
    'context': 'Контекстные фильтры',
    'chain': 'Цепочки ошибок',
    'normalization': 'Нормализация',
    'ml': 'ML-классификатор',
    'smart': 'SmartFilter',
}


# =============================================================================
# ЗАГРУЗКА GOLDEN СТАНДАРТА
# =============================================================================

def load_golden_errors(chapter_num: str) -> List[Dict]:
    """Загружает golden ошибки для главы."""
    golden_file = GOLDEN_FILES.get(chapter_num)
    if not golden_file or not golden_file.exists():
        return []

    with open(golden_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('errors', data) if isinstance(data, dict) else data


def normalize_for_comparison(word: str) -> str:
    """Нормализует слово для сравнения."""
    if not word:
        return ''
    return word.lower().strip().replace('ё', 'е')


def is_golden_error(error: Dict, golden_errors: List[Dict]) -> bool:
    """Проверяет, является ли ошибка golden."""
    error_type = error.get('type', '')

    if error_type == 'substitution':
        orig = normalize_for_comparison(error.get('original', '') or error.get('correct', ''))
        trans = normalize_for_comparison(error.get('transcript', '') or error.get('wrong', ''))
    elif error_type == 'insertion':
        orig = ''
        trans = normalize_for_comparison(error.get('transcript', '') or error.get('wrong', '') or error.get('word', ''))
    elif error_type == 'deletion':
        orig = normalize_for_comparison(error.get('original', '') or error.get('correct', '') or error.get('word', ''))
        trans = ''
    else:
        return False

    error_time = error.get('time', 0)
    # Преобразуем время в float
    try:
        error_time = float(error_time) if error_time else 0.0
    except (ValueError, TypeError):
        error_time = 0.0

    for golden in golden_errors:
        g_type = golden.get('type', '')
        if g_type != error_type:
            continue

        if g_type == 'substitution':
            g_orig = normalize_for_comparison(golden.get('original', '') or golden.get('correct', ''))
            g_trans = normalize_for_comparison(golden.get('transcript', '') or golden.get('wrong', ''))
        elif g_type == 'insertion':
            g_orig = ''
            g_trans = normalize_for_comparison(golden.get('transcript', '') or golden.get('wrong', '') or golden.get('word', ''))
        elif g_type == 'deletion':
            g_orig = normalize_for_comparison(golden.get('original', '') or golden.get('correct', '') or golden.get('word', ''))
            g_trans = ''
        else:
            continue

        if orig == g_orig and trans == g_trans:
            # Проверяем время (±5 секунд)
            g_time = golden.get('time', 0)
            try:
                g_time = float(g_time) if g_time else 0.0
            except (ValueError, TypeError):
                g_time = 0.0
            if abs(error_time - g_time) <= 5:
                return True

    return False


# =============================================================================
# АНАЛИЗ ФИЛЬТРАЦИИ
# =============================================================================

def analyze_single_filter_pass(
    errors: List[Dict],
    golden_errors: List[Dict],
) -> Dict[str, Any]:
    """
    Анализирует результат одного прогона фильтрации.

    Возвращает статистику по каждому уровню фильтрации.
    """
    from filters import should_filter_error, filter_errors
    from filters.detectors import detect_alignment_chains, detect_linked_prefix_errors

    stats = defaultdict(lambda: {
        'filtered_count': 0,
        'golden_hit': 0,
        'golden_protected': 0,
        'examples': [],
    })

    # Детектируем цепочки
    chain_indices = detect_alignment_chains(errors)
    linked_prefix_indices = detect_linked_prefix_errors(errors)

    for idx, error in enumerate(errors):
        is_golden = is_golden_error(error, golden_errors)

        # Проверяем цепочки
        if idx in chain_indices:
            stats['alignment_chain']['filtered_count'] += 1
            if is_golden:
                stats['alignment_chain']['golden_hit'] += 1
            continue

        if idx in linked_prefix_indices:
            stats['linked_prefix_error']['filtered_count'] += 1
            if is_golden:
                stats['linked_prefix_error']['golden_hit'] += 1
            continue

        # Основная фильтрация
        should_filter, reason = should_filter_error(error, all_errors=errors)

        if reason.startswith('PROTECTED_'):
            # Защитный слой — не фильтрует, а защищает
            stats[reason]['golden_protected'] += 1 if is_golden else 0
            stats[reason]['filtered_count'] += 1  # считаем как "обработано"
        elif should_filter:
            stats[reason]['filtered_count'] += 1
            if is_golden:
                stats[reason]['golden_hit'] += 1
                # Сохраняем примеры golden, которые были отфильтрованы (КРИТИЧНО!)
                stats[reason]['examples'].append({
                    'type': error.get('type'),
                    'original': error.get('original', error.get('correct', '')),
                    'transcript': error.get('transcript', error.get('wrong', '')),
                    'time': error.get('time'),
                })

    return dict(stats)


def run_full_analysis(
    transcript_path: str,
    original_path: str,
    chapter_num: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Прогоняет полный пайплайн с детальным анализом.

    1. Нормализация
    2. Умное сравнение
    3. Анализ каждого уровня фильтрации
    """
    from smart_compare import smart_compare

    # Загружаем golden
    golden_errors = load_golden_errors(chapter_num)

    # Шаг 1: Нормализация
    print(f"    Нормализация...")
    # (предполагаем, что оригинал уже нормализован или делаем это)

    # Шаг 2: Умное сравнение
    print(f"    Умное сравнение...")
    compared_path = output_dir / f'{chapter_num.zfill(2)}_analysis_compared.json'

    result = smart_compare(
        transcript_path=transcript_path,
        original_path=original_path,
        output_path=str(compared_path),
        force=True,
    )

    errors = result.get('errors', [])
    total_errors = len(errors)

    print(f"    Найдено различий: {total_errors}")

    # Шаг 3: Анализ фильтрации
    print(f"    Анализ фильтрации...")
    filter_stats = analyze_single_filter_pass(errors, golden_errors)

    # Подсчёт итогов
    total_filtered = sum(
        s['filtered_count'] for reason, s in filter_stats.items()
        if not reason.startswith('PROTECTED_')
    )
    total_golden_hit = sum(
        s['golden_hit'] for s in filter_stats.values()
    )
    total_protected = sum(
        s['golden_protected'] for s in filter_stats.values()
    )

    analysis = {
        'transcript': str(transcript_path),
        'original': str(original_path),
        'chapter': chapter_num,
        'timestamp': datetime.now().isoformat(),
        'golden_count': len(golden_errors),
        'total_differences': total_errors,
        'total_filtered': total_filtered,
        'remaining_errors': total_errors - total_filtered,
        'golden_hit': total_golden_hit,
        'golden_protected': total_protected,
        'filter_stats': filter_stats,
    }

    return analysis


def find_all_transcripts() -> List[Dict]:
    """Находит все транскрипции в проекте."""
    transcripts = []

    for chapter_dir in TRANSCRIPTIONS_DIR.iterdir():
        if not chapter_dir.is_dir() or not chapter_dir.name.startswith('Глава'):
            continue

        chapter_num = chapter_dir.name.replace('Глава', '').strip()

        for json_file in chapter_dir.glob('*.json'):
            # Пропускаем служебные файлы
            if any(x in json_file.name for x in ['_compared', '_filtered', '_analysis']):
                continue

            # Определяем тип транскрипции
            name = json_file.name
            if 'kbps' in name.lower():
                # Извлекаем битрейт
                import re
                match = re.search(r'(\d+)kbps', name, re.IGNORECASE)
                bitrate = match.group(1) if match else 'unknown'
                trans_type = f'{bitrate}kbps'
            elif 'yandex' in name.lower():
                trans_type = 'yandex'
            else:
                trans_type = 'standard'

            transcripts.append({
                'path': json_file,
                'chapter': chapter_num,
                'type': trans_type,
                'name': json_file.stem,
            })

    return sorted(transcripts, key=lambda x: (x['chapter'], x['type']))


def find_original_for_chapter(chapter_num: str) -> Optional[Path]:
    """Находит файл оригинала для главы."""
    variants = [
        f'Глава{chapter_num}.docx',
        f'Глава {chapter_num}.docx',
        f'Глава_{chapter_num}.docx',
    ]

    for variant in variants:
        path = CHAPTERS_DIR / variant
        if path.exists():
            return path

    return None


# =============================================================================
# ОТЧЁТЫ И ВИЗУАЛИЗАЦИЯ
# =============================================================================

def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Выводит отчёт по анализу."""
    print(f"\n{'='*70}")
    print(f"  АНАЛИЗ: {Path(analysis['transcript']).name}")
    print(f"{'='*70}")
    print(f"  Глава: {analysis['chapter']}")
    print(f"  Golden ошибок: {analysis['golden_count']}")
    print(f"  Всего различий: {analysis['total_differences']}")
    print(f"  Отфильтровано: {analysis['total_filtered']}")
    print(f"  Осталось: {analysis['remaining_errors']}")
    print(f"  Golden затронуто: {analysis['golden_hit']} {'⚠ КРИТИЧНО!' if analysis['golden_hit'] > 0 else '✓'}")
    print(f"  Golden защищено: {analysis['golden_protected']}")
    print(f"\n  {'Фильтр':<35} {'Отфильтр.':<12} {'Golden':<8} {'Статус'}")
    print(f"  {'-'*65}")

    stats = analysis['filter_stats']

    for reason, data in sorted(stats.items(), key=lambda x: -x[1]['filtered_count']):
        if reason.startswith('PROTECTED_'):
            status = f"защитил {data['golden_protected']}"
            count_str = f"({data['filtered_count']})"
        else:
            status = '✓' if data['golden_hit'] == 0 else f"⚠ {data['golden_hit']}"
            count_str = str(data['filtered_count'])

        print(f"  {reason:<35} {count_str:<12} {data['golden_hit']:<8} {status}")

        # Показываем примеры проблемных golden
        if data['golden_hit'] > 0 and data['examples']:
            for ex in data['examples'][:3]:
                print(f"      → {ex['original']} → {ex['transcript']} (t={ex['time']})")

    print(f"{'='*70}\n")


def build_effectiveness_matrix(analyses: List[Dict]) -> Dict[str, Any]:
    """
    Строит матрицу эффективности фильтров.

    Анализирует все прогоны и определяет:
    - Эффективные фильтры (много FP, 0 golden)
    - Бесполезные (0 или мало фильтраций)
    - Вредные (затрагивают golden)
    - Требующие доработки
    """
    matrix = defaultdict(lambda: {
        'total_filtered': 0,
        'total_golden_hit': 0,
        'total_golden_protected': 0,
        'runs': 0,
        'effectiveness': 0.0,
        'status': 'unknown',
        'examples': [],
    })

    for analysis in analyses:
        for reason, data in analysis['filter_stats'].items():
            matrix[reason]['total_filtered'] += data['filtered_count']
            matrix[reason]['total_golden_hit'] += data['golden_hit']
            matrix[reason]['total_golden_protected'] += data.get('golden_protected', 0)
            matrix[reason]['runs'] += 1
            matrix[reason]['examples'].extend(data.get('examples', []))

    # Определяем статус каждого фильтра
    for reason, data in matrix.items():
        if reason.startswith('PROTECTED_'):
            data['status'] = 'protection'
            data['effectiveness'] = data['total_golden_protected']
        elif data['total_golden_hit'] > 0:
            data['status'] = 'harmful'
            data['effectiveness'] = -data['total_golden_hit']
        elif data['total_filtered'] == 0:
            data['status'] = 'useless'
            data['effectiveness'] = 0
        elif data['total_filtered'] < 5:
            data['status'] = 'low_impact'
            data['effectiveness'] = data['total_filtered']
        else:
            data['status'] = 'effective'
            data['effectiveness'] = data['total_filtered']

    return dict(matrix)


def print_effectiveness_matrix(matrix: Dict[str, Any]) -> None:
    """Выводит матрицу эффективности."""
    print(f"\n{'#'*70}")
    print(f"  МАТРИЦА ЭФФЕКТИВНОСТИ ФИЛЬТРОВ")
    print(f"{'#'*70}")

    # Группируем по статусу
    by_status = defaultdict(list)
    for reason, data in matrix.items():
        by_status[data['status']].append((reason, data))

    status_order = ['harmful', 'protection', 'effective', 'low_impact', 'useless']
    status_labels = {
        'harmful': '⚠ ВРЕДНЫЕ (затрагивают Golden)',
        'protection': '🛡 ЗАЩИТНЫЕ (предотвращают фильтрацию Golden)',
        'effective': '✓ ЭФФЕКТИВНЫЕ (фильтруют FP без потерь)',
        'low_impact': '○ НИЗКИЙ ЭФФЕКТ (< 5 фильтраций)',
        'useless': '✗ БЕСПОЛЕЗНЫЕ (0 фильтраций)',
    }

    for status in status_order:
        items = by_status.get(status, [])
        if not items:
            continue

        print(f"\n  {status_labels[status]}")
        print(f"  {'-'*60}")

        for reason, data in sorted(items, key=lambda x: -abs(x[1]['effectiveness'])):
            if status == 'protection':
                print(f"    {reason:<35} защитил {data['total_golden_protected']} golden")
            else:
                print(f"    {reason:<35} {data['total_filtered']:>4} FP, {data['total_golden_hit']:>2} golden")
                if data['total_golden_hit'] > 0:
                    for ex in data['examples'][:2]:
                        print(f"        → {ex['original']} → {ex['transcript']}")

    print(f"\n{'#'*70}\n")


def save_analysis(analysis: Dict[str, Any], output_dir: Path) -> Path:
    """Сохраняет анализ в файл."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_name = Path(analysis['transcript']).stem
    filename = f"{transcript_name}_analysis.json"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'Анализ эффективности фильтров v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--chapter', '-c', choices=['1', '2', '3', '4'],
                        help='Анализировать только указанную главу')
    parser.add_argument('--transcript', '-t', type=str,
                        help='Путь к конкретной транскрипции')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Показать сводку по всем прогонам')
    parser.add_argument('--matrix', '-m', action='store_true',
                        help='Построить матрицу эффективности')
    parser.add_argument('--skip-bitrate', action='store_true',
                        help='Пропустить транскрипции с битрейтом (только основные)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')
    parser.add_argument('--version', '-V', action='store_true',
                        help='Показать версию')

    args = parser.parse_args()

    if args.version:
        print(f"filter_analysis v{VERSION} ({VERSION_DATE})")
        return 0

    # Создаём папку для результатов
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем существующие анализы для матрицы
    if args.summary or args.matrix:
        analyses = []
        for json_file in ANALYSIS_DIR.glob('*_analysis.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    analyses.append(json.load(f))
            except Exception as e:
                print(f"Ошибка загрузки {json_file.name}: {e}")

        if not analyses:
            print("Нет данных для анализа. Сначала запустите прогон транскрипций.")
            return 1

        if args.matrix:
            matrix = build_effectiveness_matrix(analyses)
            print_effectiveness_matrix(matrix)

            # Сохраняем матрицу
            matrix_path = ANALYSIS_DIR / 'effectiveness_matrix.json'
            with open(matrix_path, 'w', encoding='utf-8') as f:
                json.dump(matrix, f, ensure_ascii=False, indent=2, default=str)
            print(f"Матрица сохранена: {matrix_path}")

        if args.summary:
            print(f"\n{'='*70}")
            print(f"  СВОДКА ПО {len(analyses)} ПРОГОНАМ")
            print(f"{'='*70}")

            for analysis in sorted(analyses, key=lambda x: (x['chapter'], x['transcript'])):
                trans_name = Path(analysis['transcript']).name
                status = '✓' if analysis['golden_hit'] == 0 else '⚠'
                print(f"  {status} Гл.{analysis['chapter']} {trans_name:<40} "
                      f"FP: {analysis['remaining_errors']:>3} / {analysis['total_differences']:>3}")

            print(f"{'='*70}\n")

        return 0

    # Находим транскрипции для анализа
    if args.transcript:
        transcripts = [{'path': Path(args.transcript), 'chapter': '1', 'type': 'manual'}]
    else:
        transcripts = find_all_transcripts()

        if args.chapter:
            transcripts = [t for t in transcripts if t['chapter'] == args.chapter]

        if args.skip_bitrate:
            transcripts = [t for t in transcripts if 'kbps' not in t['type']]

    if not transcripts:
        print("Не найдено транскрипций для анализа.")
        return 1

    print(f"\n{'#'*70}")
    print(f"  АНАЛИЗ ЭФФЕКТИВНОСТИ ФИЛЬТРОВ v{VERSION}")
    print(f"{'#'*70}")
    print(f"  Транскрипций для анализа: {len(transcripts)}")
    print(f"  Папка результатов: {ANALYSIS_DIR}")
    print(f"{'#'*70}\n")

    analyses = []

    for trans in transcripts:
        print(f"\n  Анализ: {trans['path'].name}")
        print(f"  Глава: {trans['chapter']}, Тип: {trans['type']}")

        # Находим оригинал
        original = find_original_for_chapter(trans['chapter'])
        if not original:
            print(f"    ⚠ Не найден оригинал для главы {trans['chapter']}")
            continue

        try:
            analysis = run_full_analysis(
                transcript_path=str(trans['path']),
                original_path=str(original),
                chapter_num=trans['chapter'],
                output_dir=ANALYSIS_DIR,
            )

            print_analysis_report(analysis)

            save_path = save_analysis(analysis, ANALYSIS_DIR)
            print(f"  Сохранено: {save_path.name}")

            analyses.append(analysis)

        except Exception as e:
            print(f"    ✗ Ошибка: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Итоговая матрица
    if len(analyses) > 0:
        matrix = build_effectiveness_matrix(analyses)
        print_effectiveness_matrix(matrix)

        # Сохраняем матрицу
        matrix_path = ANALYSIS_DIR / 'effectiveness_matrix.json'
        with open(matrix_path, 'w', encoding='utf-8') as f:
            json.dump(matrix, f, ensure_ascii=False, indent=2, default=str)
        print(f"Матрица сохранена: {matrix_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
