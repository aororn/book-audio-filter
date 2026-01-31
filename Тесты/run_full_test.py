#!/usr/bin/env python3
"""
Полный тест золотого стандарта v6.1

Прогоняет пайплайн проверки для глав 1-4, затем сверяет результаты
с эталонными наборами ошибок (золотыми стандартами).

Использует актуальный pipeline.py + config.py.

Новое в v6.1:
- Динамический автопоиск файлов транскрипций и оригиналов
- Больше никаких захардкоженных путей с датами
- Функции find_transcript_file(), find_original_file()

Новое в v6.0:
- Автоматическое определение версии фильтров из engine.py
- Сохранение каждого прогона в отдельный файл в папке История/
- Поддержка главы 4
- Сводная таблица по версиям

Использование:
    python Тесты/run_full_test.py              # все главы
    python Тесты/run_full_test.py --chapter 1   # только глава 1
    python Тесты/run_full_test.py --skip-pipeline  # только золотой тест (без пайплайна)
    python Тесты/run_full_test.py --clean        # чистое тестирование (удалить старые файлы)
    python Тесты/run_full_test.py --verbose      # подробный вывод
    python Тесты/run_full_test.py --no-log       # без записи в историю
    python Тесты/run_full_test.py --versions     # сводка по версиям

Changelog:
    v6.3 (2026-01-30): Унификация версионирования
        - Добавлена запись в единый versions.json при успешном полном тесте
        - update_versions_json() — функция обновления источника правды
        - Интеграция с version_table.py для быстрого сравнения версий
    v6.2 (2026-01-29): Флаг --clean для чистого тестирования
        - Добавлен флаг --clean — удаляет старые compared/filtered перед пайплайном
        - Это гарантирует тестирование на свежих файлах, а не на кэшированных
        - clean_chapter_results() — функция очистки результатов главы
    v6.1 (2026-01-29): Динамический автопоиск файлов
        - find_transcript_file() — поиск транскрипции
        - find_original_file() — поиск оригинала
        - Удалены захардкоженные пути с датами
    v6.0 (2026-01-29): Архивирование прогонов
    v5.2 (2026-01-25): Сравнение с историей
    v5.1 (2026-01-25): Логирование и статистика
"""

VERSION = '6.3.0'
VERSION_DATE = '2026-01-30'

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Добавляем путь к модулям
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'Инструменты'))

from config import (
    RESULTS_DIR, TESTS_DIR, TRANSCRIPTIONS_DIR,
    CHAPTERS_DIR, AUDIO_DIR, FileNaming
)
from test_golden_standard import test_golden_standard
from version import PROJECT_VERSION, FILTER_ENGINE_VERSION, SMART_COMPARE_VERSION


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

HISTORY_FILE = TESTS_DIR / 'golden_test_history.json'
HISTORY_DIR = TESTS_DIR / 'История'
VERSIONS_FILE = TESTS_DIR / 'versions.json'


def clean_chapter_results(chapter_cfg) -> int:
    """
    Удаляет все compared и filtered файлы для главы.

    Возвращает количество удалённых файлов.

    Это нужно для чистого тестирования — чтобы пайплайн создавал
    новые файлы текущей версией smart_compare.py, а не использовал
    закэшированные файлы от старых версий.
    """
    results_dir = chapter_cfg['results_dir']
    chapter_id = chapter_cfg['chapter_id']
    deleted = 0

    if not results_dir.exists():
        return 0

    # Паттерны файлов для удаления
    patterns = [
        f'{chapter_id}*_compared.json',
        f'{chapter_id}*_filtered.json',
    ]

    for pattern in patterns:
        for filepath in results_dir.glob(pattern):
            try:
                filepath.unlink()
                deleted += 1
                print(f"    Удалён: {filepath.name}")
            except Exception as e:
                print(f"    Ошибка удаления {filepath.name}: {e}")

    return deleted


def find_transcript_file(chapter_id: str) -> Path | None:
    """
    Автоматический поиск файла транскрипции для главы.

    Приоритет:
    1. Результаты проверки/{chapter_id}/{chapter_id}_transcript.json
    2. Транскрипции/Глава{N}/{chapter_id}_transcript.json
    3. {chapter_id}_yandex_transcript*.json (старое имя)
    4. Файл с "transcript" и самой поздней датой (NEW_YYYYMMDD)
    5. Любой JSON без _compared/_filtered

    Исключаем файлы с битрейтом (16kbps, 32kbps и т.д.) — это тестовые.
    """
    chapter_num = chapter_id.lstrip('0') or '0'

    # 0. Проверяем в папке результатов (новый формат)
    results_transcript = RESULTS_DIR / chapter_id / f'{chapter_id}_transcript.json'
    if results_transcript.exists():
        return results_transcript

    trans_dir = TRANSCRIPTIONS_DIR / f'Глава{chapter_num}'

    if not trans_dir.exists():
        return None

    # 1. Стандартное имя
    standard = trans_dir / f'{chapter_id}_transcript.json'
    if standard.exists():
        return standard

    # 2. Старое имя yandex_transcript
    yandex_files = list(trans_dir.glob(f'{chapter_id}_yandex_transcript*.json'))
    if yandex_files:
        # Берём самый новый по дате в имени
        return sorted(yandex_files, reverse=True)[0]

    # 3. Любой файл с "transcript", но без битрейта
    candidates = list(trans_dir.glob(f'*transcript*.json'))
    # Исключаем файлы с битрейтом (16kbps, 32kbps, etc.)
    candidates = [f for f in candidates if 'kbps' not in f.name.lower()]
    if candidates:
        # Сортируем по дате в имени (NEW_YYYYMMDD) — новые первые
        return sorted(candidates, reverse=True)[0]

    # 4. Первый JSON в папке (кроме служебных)
    all_json = list(trans_dir.glob('*.json'))
    all_json = [f for f in all_json
                if '_compared' not in f.name
                and '_filtered' not in f.name
                and 'kbps' not in f.name.lower()]
    if all_json:
        return sorted(all_json, reverse=True)[0]

    return None


def find_original_file(chapter_id: str) -> Path | None:
    """
    Автоматический поиск файла оригинала для главы.

    Поиск: Глава{N}.docx, Глава {N}.docx, Глава_{N}.docx, Глава{N}.txt
    """
    chapter_num = chapter_id.lstrip('0') or '0'

    # Варианты именования (сначала docx, затем txt)
    variants = [
        f'Глава{chapter_num}.docx',
        f'Глава {chapter_num}.docx',
        f'Глава_{chapter_num}.docx',
        f'Глава{chapter_num}.txt',
        f'Глава {chapter_num}.txt',
        f'Глава_{chapter_num}.txt',
    ]

    for variant in variants:
        path = CHAPTERS_DIR / variant
        if path.exists():
            return path

    return None


def get_chapter_config(chapter_num: str) -> dict:
    """
    Динамическая генерация конфигурации главы.

    Использует автопоиск файлов вместо захардкоженных путей.
    """
    chapter_id = chapter_num.zfill(2)  # '1' -> '01'

    return {
        'chapter_id': chapter_id,
        'audio': AUDIO_DIR / f'{chapter_id}.mp3',
        'original': find_original_file(chapter_id),
        'transcript': find_transcript_file(chapter_id),
        'golden_standard': TESTS_DIR / f'золотой_стандарт_глава{chapter_num}.json',
        'results_dir': RESULTS_DIR / chapter_id,
    }


# Динамическая конфигурация глав (автопоиск файлов)
CHAPTERS = {str(i): get_chapter_config(str(i)) for i in range(1, 10)}


def get_project_version():
    """Возвращает версию проекта из version.py"""
    return PROJECT_VERSION


def get_filter_version():
    """Возвращает версию движка фильтрации из version.py"""
    return FILTER_ENGINE_VERSION


def get_smart_compare_version():
    """Возвращает версию smart_compare из version.py"""
    return SMART_COMPARE_VERSION


def check_chapter_files(chapter_cfg):
    """Проверяет наличие файлов для главы."""
    missing = []
    for key in ('audio', 'original', 'transcript', 'golden_standard'):
        path = chapter_cfg[key]
        if path is None or not path.exists():
            missing.append(f"  {key}: {path}")
    return missing


def run_pipeline_for_chapter(chapter_cfg, force=False, verbose=False):
    """Запускает пайплайн для одной главы."""
    from pipeline import run_pipeline

    results = run_pipeline(
        audio_path=str(chapter_cfg['audio']),
        text_path=str(chapter_cfg['original']),
        output_dir=str(chapter_cfg['results_dir']),
        skip_transcribe=True,
        transcript_path=str(chapter_cfg['transcript']),
        force=force,
        show_progress=False,
    )
    return results


def find_filtered_file(chapter_cfg):
    """Ищет самый свежий filtered файл для главы."""
    chapter_id = chapter_cfg['chapter_id']
    results_dir = chapter_cfg['results_dir']

    # Ищем файлы по паттерну *_filtered.json
    filtered_files = list(results_dir.glob(f'{chapter_id}*_filtered.json'))

    if not filtered_files:
        return None

    # Возвращаем самый новый файл
    return max(filtered_files, key=lambda p: p.stat().st_mtime)


def get_total_errors_count(chapter_cfg):
    """Получает общее количество ошибок из отфильтрованного отчёта."""
    filtered_path = find_filtered_file(chapter_cfg)

    if not filtered_path or not filtered_path.exists():
        return None

    try:
        with open(filtered_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return len(report.get('errors', []))
    except Exception:
        return None


def run_golden_test_for_chapter(chapter_num, chapter_cfg, verbose=True):
    """Запускает золотой тест для одной главы."""
    filtered_path = find_filtered_file(chapter_cfg)

    if not filtered_path or not filtered_path.exists():
        print(f"  Отчёт не найден в: {chapter_cfg['results_dir']}")
        return None, 0, 0, [], None

    # Получаем общее количество ошибок
    total_errors = get_total_errors_count(chapter_cfg)

    passed, found, total, missing = test_golden_standard(
        str(filtered_path),
        str(chapter_cfg['golden_standard']),
        verbose=verbose
    )
    return passed, found, total, missing, total_errors


def load_history():
    """Загружает историю тестов."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'runs': []}
    return {'runs': []}


def get_chapter_history(chapter_num, num_runs=4):
    """
    Получает историю количества ошибок для главы за последние N запусков.
    """
    history = load_history()
    runs = history.get('runs', [])

    chapter_key = f'chapter_{chapter_num}'
    errors_history = []

    for run in reversed(runs):
        if chapter_key in run.get('results', {}):
            total_errors = run['results'][chapter_key].get('total_errors', 0)
            errors_history.append(total_errors)
            if len(errors_history) >= num_runs:
                break

    while len(errors_history) < num_runs:
        errors_history.append(None)

    return errors_history


def get_total_history(num_runs=4):
    """Получает историю общего количества ошибок за последние N запусков."""
    history = load_history()
    runs = history.get('runs', [])

    errors_history = []
    for run in reversed(runs):
        total_errors = run.get('summary', {}).get('total_errors', 0)
        errors_history.append(total_errors)
        if len(errors_history) >= num_runs:
            break

    while len(errors_history) < num_runs:
        errors_history.append(None)

    return errors_history


def format_trend(current, previous):
    """Форматирует тренд изменения."""
    if previous is None or current is None:
        return ''
    diff = current - previous
    if diff < 0:
        return f'↓{abs(diff)}'
    elif diff > 0:
        return f'↑{diff}'
    else:
        return '='


def format_history_cell(value):
    """Форматирует ячейку истории."""
    if value is None:
        return '-'
    return str(value)


def save_history(history):
    """Сохраняет историю тестов в главный файл."""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_run_to_archive(run_entry):
    """
    Сохраняет прогон в отдельный файл в папке История/.
    Формат имени: YYYY-MM-DD_HH-MM_vX.Y.Z_golden.json
    """
    HISTORY_DIR.mkdir(exist_ok=True)

    ts = datetime.now()
    date_str = ts.strftime('%Y-%m-%d_%H-%M')
    project_ver = run_entry.get('project_version', get_project_version())

    filename = f"{date_str}_v{project_ver}_golden.json"
    filepath = HISTORY_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(run_entry, f, ensure_ascii=False, indent=2)

    return filepath


def log_test_run(results, chapters_tested, comment=None):
    """Записывает результат теста в историю."""
    history = load_history()

    project_version = get_project_version()
    filter_version = get_filter_version()
    smart_compare_version = get_smart_compare_version()

    run_entry = {
        'timestamp': datetime.now().isoformat(),
        'test_version': VERSION,
        'project_version': project_version,
        'filter_version': filter_version,
        'smart_compare_version': smart_compare_version,
        'chapters_tested': chapters_tested,
        'results': {},
        'summary': {
            'total_golden_found': 0,
            'total_golden_expected': 0,
            'total_errors': 0,
            'all_passed': True,
        }
    }

    if comment:
        run_entry['comment'] = comment

    for ch_num, res in results.items():
        run_entry['results'][f'chapter_{ch_num}'] = {
            'passed': res['passed'],
            'golden_found': res['found'],
            'golden_expected': res['total'],
            'total_errors': res.get('total_errors', 0),
        }
        run_entry['summary']['total_golden_found'] += res['found']
        run_entry['summary']['total_golden_expected'] += res['total']
        run_entry['summary']['total_errors'] += res.get('total_errors', 0) or 0
        if not res['passed']:
            run_entry['summary']['all_passed'] = False

    # Вычисляем процент
    if run_entry['summary']['total_golden_expected'] > 0:
        run_entry['summary']['golden_percentage'] = round(
            100 * run_entry['summary']['total_golden_found'] /
            run_entry['summary']['total_golden_expected'], 1
        )

    # Сохраняем в главный файл истории
    history['runs'].append(run_entry)
    save_history(history)

    # Сохраняем в отдельный файл архива
    archive_path = save_run_to_archive(run_entry)

    return run_entry, archive_path


def update_versions_json(run_entry: dict, version_id: str = None) -> None:
    """
    Обновляет versions.json с текущими результатами.

    Это ЕДИНЫЙ источник правды для сравнения версий.

    Логика определения version_id:
    1. Явно переданный version_id
    2. Из project_version (PROJECT_VERSION из version.py)
    """
    if not VERSIONS_FILE.exists():
        versions_data = {"_schema_version": "1.0", "_description": "Единый источник истины для версий проекта и метрик", "versions": []}
    else:
        with open(VERSIONS_FILE, 'r', encoding='utf-8') as f:
            versions_data = json.load(f)

    project_ver = run_entry.get('project_version', get_project_version())
    filter_ver = run_entry.get('filter_version', '?')
    smart_compare_ver = run_entry.get('smart_compare_version', '?')

    if not version_id:
        # Используем project_version как ID
        version_id = f"v{project_ver}"

    # Формируем данные о версии
    results = run_entry.get('results', {})
    summary = run_entry.get('summary', {})

    chapters = {}
    for ch_key, ch_data in results.items():
        ch_num = ch_key.replace('chapter_', '')
        chapters[ch_num] = {
            'total': ch_data.get('total_errors', 0),
            'golden': ch_data.get('golden_expected', 0)
        }

    new_version = {
        'id': version_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'project_version': project_ver,
        'filter_version': filter_ver,
        'smart_compare_version': smart_compare_ver,
        'chapters': chapters,
        'totals': {
            'errors': summary.get('total_errors', 0),
            'golden': summary.get('total_golden_expected', 0),
            'fp': summary.get('total_errors', 0) - summary.get('total_golden_expected', 0)
        },
        'notes': run_entry.get('comment', f'Auto-saved by run_full_test.py v{VERSION}')
    }

    # Ищем существующую версию с таким ID
    existing_idx = None
    for i, v in enumerate(versions_data['versions']):
        if v['id'] == version_id:
            existing_idx = i
            break

    if existing_idx is not None:
        # Обновляем существующую
        versions_data['versions'][existing_idx] = new_version
    else:
        # Добавляем новую
        versions_data['versions'].append(new_version)

    versions_data['_updated'] = datetime.now().strftime('%Y-%m-%d')

    with open(VERSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(versions_data, f, ensure_ascii=False, indent=2)


def show_versions_summary():
    """Показывает сводку результатов по версиям фильтров."""
    history = load_history()
    runs = history.get('runs', [])

    if not runs:
        print("История пуста.")
        return

    # Группируем по версии фильтров
    by_version = {}
    for run in runs:
        ver = run.get('filter_version', run.get('version', 'unknown'))
        if ver not in by_version:
            by_version[ver] = []
        by_version[ver].append(run)

    print(f"\n{'='*80}")
    print(f"  СВОДКА ПО ВЕРСИЯМ ФИЛЬТРОВ")
    print(f"{'='*80}")

    # Заголовок
    print(f"\n  {'Версия':<12} {'Прогонов':<10} {'Golden':<12} {'FP (мин)':<10} {'FP (макс)':<10} {'Последний'}")
    print(f"  {'-'*72}")

    for ver in sorted(by_version.keys(), key=lambda x: x if x != 'unknown' else 'zzz'):
        ver_runs = by_version[ver]

        # Статистика
        num_runs = len(ver_runs)
        golden_results = []
        fp_results = []

        for r in ver_runs:
            s = r.get('summary', {})
            golden_str = f"{s.get('total_golden_found', 0)}/{s.get('total_golden_expected', 0)}"
            golden_results.append(golden_str)
            fp_results.append(s.get('total_errors', 0))

        # Последний прогон
        last_run = ver_runs[-1]
        last_ts = last_run['timestamp'][:16].replace('T', ' ')
        last_golden = f"{last_run['summary'].get('total_golden_found', 0)}/{last_run['summary'].get('total_golden_expected', 0)}"

        fp_min = min(fp_results) if fp_results else '-'
        fp_max = max(fp_results) if fp_results else '-'

        print(f"  v{ver:<11} {num_runs:<10} {last_golden:<12} {fp_min:<10} {fp_max:<10} {last_ts}")

    print(f"\n{'='*80}\n")


def show_archive_files():
    """Показывает список файлов в архиве."""
    if not HISTORY_DIR.exists():
        print("Папка История/ не существует.")
        return

    files = sorted(HISTORY_DIR.glob('*.json'), reverse=True)

    if not files:
        print("Архив пуст.")
        return

    print(f"\n{'='*70}")
    print(f"  АРХИВ ПРОГОНОВ ({len(files)} файлов)")
    print(f"{'='*70}")

    for f in files[:20]:  # Показываем последние 20
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            s = data.get('summary', {})
            golden = f"{s.get('total_golden_found', 0)}/{s.get('total_golden_expected', 0)}"
            fp = s.get('total_errors', 0)
            passed = '✓' if s.get('all_passed', False) else '✗'
            comment = data.get('comment', '')[:30]
            print(f"  {passed} {f.name:<45} Golden: {golden:<8} FP: {fp:<4} {comment}")
        except Exception as e:
            print(f"  ? {f.name:<45} (ошибка чтения)")

    if len(files) > 20:
        print(f"\n  ... и ещё {len(files) - 20} файлов")

    print(f"\n{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'Полный тест золотого стандарта v{VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python Тесты/run_full_test.py              # все главы
  python Тесты/run_full_test.py --chapter 1   # только глава 1
  python Тесты/run_full_test.py --skip-pipeline  # только золотой тест
  python Тесты/run_full_test.py --clean        # чистое тестирование
  python Тесты/run_full_test.py --versions     # сводка по версиям
  python Тесты/run_full_test.py --archive      # список файлов архива
        """
    )
    parser.add_argument('--chapter', '-c', choices=['1', '2', '3', '4', '5'],
                        help='Номер главы (по умолчанию: все)')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Пропустить пайплайн, запустить только золотой тест')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Перезаписать существующие результаты')
    parser.add_argument('--clean', action='store_true',
                        help='Удалить старые compared/filtered файлы перед запуском (чистое тестирование)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод')
    parser.add_argument('--no-log', action='store_true',
                        help='Не записывать результат в историю')
    parser.add_argument('--comment', '-m', type=str,
                        help='Комментарий к запуску (для истории)')
    parser.add_argument('--version', '-V', action='store_true',
                        help='Показать версию')
    parser.add_argument('--history', '-H', action='store_true',
                        help='Показать последние 10 записей истории')
    parser.add_argument('--versions', action='store_true',
                        help='Показать сводку по версиям фильтров')
    parser.add_argument('--archive', action='store_true',
                        help='Показать список файлов архива')

    args = parser.parse_args()

    if args.version:
        print(f"run_full_test v{VERSION} ({VERSION_DATE})")
        print(f"  Filter version: {get_filter_version()}")
        print(f"  SmartCompare version: {get_smart_compare_version()}")
        return 0

    if args.versions:
        show_versions_summary()
        return 0

    if args.archive:
        show_archive_files()
        return 0

    if args.history:
        history = load_history()
        runs = history.get('runs', [])[-10:]
        if not runs:
            print("История пуста.")
            return 0

        print(f"\n{'='*70}")
        print(f"  ИСТОРИЯ ТЕСТОВ (последние {len(runs)} записей)")
        print(f"{'='*70}")

        for run in runs:
            ts = run['timestamp'][:19].replace('T', ' ')
            summary = run['summary']
            status = '✓' if summary['all_passed'] else '✗'
            golden = f"{summary['total_golden_found']}/{summary['total_golden_expected']}"
            total_err = summary.get('total_errors', '?')
            pct = summary.get('golden_percentage', 0)
            filter_ver = run.get('filter_version', run.get('version', '?'))
            comment = run.get('comment', '')

            print(f"\n  {ts}  {status}  v{filter_ver}  Golden: {golden} ({pct}%)  FP: {total_err}")
            if comment:
                print(f"    Комментарий: {comment}")

        print(f"\n{'='*70}\n")
        return 0

    # Определяем какие главы тестировать
    if args.chapter:
        chapters_to_test = [args.chapter]
    else:
        chapters_to_test = ['1', '2', '3', '4', '5']

    project_ver = get_project_version()
    filter_ver = get_filter_version()
    sc_ver = get_smart_compare_version()

    print(f"\n{'#'*70}")
    print(f"  ПОЛНЫЙ ТЕСТ ЗОЛОТОГО СТАНДАРТА v{VERSION}")
    print(f"{'#'*70}")
    print(f"  Версия проекта: {project_ver}")
    print(f"  Главы: {', '.join(chapters_to_test)}")
    print(f"  Пайплайн: {'пропуск' if args.skip_pipeline else 'запуск'}")
    print(f"  Чистое тестирование: {'ДА (--clean)' if args.clean else 'нет'}")
    print(f"  Версия фильтров: {filter_ver}")
    print(f"  Версия SmartCompare: {sc_ver}")
    print(f"  Логирование: {'отключено' if args.no_log else 'включено'}")
    print(f"{'#'*70}")

    # Проверяем файлы
    valid_chapters = []
    for ch_num in chapters_to_test:
        cfg = CHAPTERS[ch_num]
        missing = check_chapter_files(cfg)
        if missing:
            print(f"\n  Глава {ch_num} — не хватает файлов:")
            for m in missing:
                print(f"    {m}")
            print(f"  Пропускаю главу {ch_num}.")
        else:
            valid_chapters.append(ch_num)

    if not valid_chapters:
        print("\n  Нет глав для тестирования.")
        return 1

    # Чистка старых файлов (если --clean)
    if args.clean and not args.skip_pipeline:
        print(f"\n{'='*70}")
        print(f"  ОЧИСТКА СТАРЫХ ФАЙЛОВ (--clean)")
        print(f"{'='*70}")
        total_deleted = 0
        for ch_num in valid_chapters:
            cfg = CHAPTERS[ch_num]
            print(f"\n  Глава {ch_num}:")
            deleted = clean_chapter_results(cfg)
            total_deleted += deleted
            if deleted == 0:
                print(f"    (нет файлов для удаления)")
        print(f"\n  Итого удалено: {total_deleted} файлов")

    # Запуск пайплайна
    if not args.skip_pipeline:
        for ch_num in valid_chapters:
            cfg = CHAPTERS[ch_num]
            print(f"\n{'='*70}")
            print(f"  ПАЙПЛАЙН: Глава {ch_num}")
            print(f"{'='*70}")
            try:
                run_pipeline_for_chapter(cfg, force=args.force, verbose=args.verbose)
            except Exception as e:
                print(f"\n  Ошибка пайплайна главы {ch_num}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return 1

    # Золотые тесты
    results = {}
    all_passed = True

    for ch_num in valid_chapters:
        cfg = CHAPTERS[ch_num]
        print(f"\n{'='*70}")
        print(f"  ЗОЛОТОЙ ТЕСТ: Глава {ch_num}")
        print(f"{'='*70}")

        passed, found, total, missing, total_errors = run_golden_test_for_chapter(
            ch_num, cfg, verbose=True
        )

        if passed is None:
            all_passed = False
            results[ch_num] = {'passed': False, 'found': 0, 'total': 0, 'total_errors': 0}
        else:
            results[ch_num] = {
                'passed': passed,
                'found': found,
                'total': total,
                'total_errors': total_errors or 0
            }
            if not passed:
                all_passed = False

    # Итоговый результат с историей
    print(f"\n{'#'*70}")
    print(f"  ИТОГИ v{project_ver} (с историей последних 4 итераций)")
    print(f"{'#'*70}")

    # Заголовок таблицы
    print(f"\n  {'Глава':<8} {'Golden':<12} {'Сейчас':<8} {'Пред.':<8} {'-2':<8} {'-3':<8} {'Тренд':<8}")
    print(f"  {'-'*60}")

    total_found = 0
    total_expected = 0
    total_all_errors = 0

    for ch_num, res in results.items():
        status = '✓' if res['passed'] else '✗'
        pct = (100 * res['found'] / res['total']) if res['total'] > 0 else 0
        err_count = res.get('total_errors', 0)
        total_all_errors += err_count

        # Получаем историю для главы
        hist = get_chapter_history(ch_num, num_runs=4)

        golden_str = f"{res['found']}/{res['total']}"
        current_str = str(err_count)
        prev_str = format_history_cell(hist[0])
        prev2_str = format_history_cell(hist[1])
        prev3_str = format_history_cell(hist[2])

        # Тренд: сравниваем текущий с предыдущим
        trend = format_trend(err_count, hist[0])

        print(f"  {status} Гл.{ch_num:<4} {golden_str:<12} {current_str:<8} {prev_str:<8} {prev2_str:<8} {prev3_str:<8} {trend:<8}")

        total_found += res['found']
        total_expected += res['total']

    # Итоговая строка
    print(f"  {'-'*60}")

    if total_expected > 0:
        total_pct = 100 * total_found / total_expected
        total_golden_str = f"{total_found}/{total_expected}"

        # История общего количества
        total_hist = get_total_history(num_runs=4)
        total_prev_str = format_history_cell(total_hist[0])
        total_prev2_str = format_history_cell(total_hist[1])
        total_prev3_str = format_history_cell(total_hist[2])
        total_trend = format_trend(total_all_errors, total_hist[0])

        print(f"  {'ИТОГО':<8} {total_golden_str:<12} {total_all_errors:<8} {total_prev_str:<8} {total_prev2_str:<8} {total_prev3_str:<8} {total_trend:<8}")

    if all_passed:
        print(f"\n  ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print(f"\n  ✗ ЕСТЬ НЕПРОЙДЕННЫЕ ТЕСТЫ")

    print(f"{'#'*70}\n")

    # Логирование
    if not args.no_log:
        run_entry, archive_path = log_test_run(results, valid_chapters, comment=args.comment)
        # Обновляем единый versions.json (источник правды для сравнения версий)
        if all_passed and len(valid_chapters) >= 4:
            update_versions_json(run_entry)
        print(f"  Результат записан в {HISTORY_FILE.name}")
        print(f"  Архив: {archive_path.name}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
