#!/usr/bin/env python3
"""
Batch Pipeline v2.0 - Пакетная обработка нескольких глав аудиокниги

Автоматически находит и обрабатывает все главы в указанной папке.
Сопоставляет аудиофайлы с текстовыми файлами по имени.

Использование:
    python batch_pipeline.py --audio-dir Главы/аудио --text-dir Главы/тексты
    python batch_pipeline.py --audio-dir . --text-dir . --pattern "глава*.mp3"
    python batch_pipeline.py --config batch_config.json
    python batch_pipeline.py --audio-dir . --text-dir . --force

Changelog:
    v2.0 (2026-01-24): Интеграция с config.py
        - Использование RESULTS_DIR, CHAPTERS_DIR из config.py
        - FileNaming для поиска транскрипций
        - format_duration для красивого вывода времени
        - check_file_exists() + флаг --force
        - VERSION/VERSION_DATE константы
    v1.0: Базовая версия пакетной обработки
"""

# Версия модуля
VERSION = '5.0.0'
VERSION_DATE = '2026-01-25'

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Добавляем путь к модулям
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# ИМПОРТ ЦЕНТРАЛИЗОВАННОЙ КОНФИГУРАЦИИ
# =============================================================================

try:
    from config import (
        PROJECT_DIR, RESULTS_DIR, CHAPTERS_DIR, AUDIO_DIR,
        FileNaming, check_file_exists, format_duration
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Fallback значения
    PROJECT_DIR = SCRIPT_DIR.parent
    RESULTS_DIR = PROJECT_DIR / 'Результаты проверки'
    ORIGINAL_DIR = PROJECT_DIR / 'Оригинал'
    CHAPTERS_DIR = ORIGINAL_DIR / 'Главы'
    AUDIO_DIR = ORIGINAL_DIR / 'Аудио'

    def format_duration(seconds):
        """Fallback форматирование времени."""
        if seconds < 0:
            return "неизвестно"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}ч {minutes:02d}м {secs:02d}с"
        return f"{minutes}м {secs:02d}с"

    def check_file_exists(path, action='skip'):
        """Fallback проверка существования файла."""
        if not path.exists():
            return True
        if action == 'overwrite':
            return True
        print(f"  → Файл уже существует: {path.name}")
        return action != 'skip'

# Опционально импортируем tqdm для прогресс-бара
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


@dataclass
class ChapterResult:
    """Результат обработки одной главы"""
    chapter_name: str
    audio_file: str
    text_file: str
    status: str = 'pending'  # pending, processing, completed, failed, skipped
    errors_found: int = 0
    errors_filtered: int = 0
    real_errors: int = 0
    output_dir: Optional[str] = None
    report_file: Optional[str] = None
    docx_file: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class BatchResult:
    """Результат пакетной обработки"""
    total_chapters: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_errors_found: int = 0
    total_real_errors: int = 0
    total_duration_seconds: float = 0.0
    chapters: List[ChapterResult] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def find_chapter_pairs(
    audio_dir: str,
    text_dir: str,
    audio_pattern: str = '*.mp3',
    text_pattern: str = '*.docx'
) -> List[Tuple[Path, Path]]:
    """
    Находит пары аудио + текст по имени файла.

    Пытается сопоставить файлы по номеру главы или по совпадению имени.
    Например: глава_01.mp3 <-> глава_01.docx
    """
    audio_dir = Path(audio_dir)
    text_dir = Path(text_dir)

    # Находим все аудио и текстовые файлы
    audio_files = list(audio_dir.glob(audio_pattern))
    text_files = list(text_dir.glob(text_pattern))

    # Также ищем .txt файлы
    text_files.extend(text_dir.glob(text_pattern.replace('.docx', '.txt')))

    print(f"  Найдено аудиофайлов: {len(audio_files)}")
    print(f"  Найдено текстовых файлов: {len(text_files)}")

    pairs = []

    def extract_number(filename: str) -> Optional[int]:
        """Извлекает номер главы из имени файла"""
        # Паттерны: глава_01, chapter01, 01_глава, etc.
        patterns = [
            r'глава[_\s]*(\d+)',
            r'chapter[_\s]*(\d+)',
            r'^(\d+)[_\s]',
            r'[_\s](\d+)$',
            r'(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def get_base_name(path: Path) -> str:
        """Получает базовое имя без расширения и номера"""
        name = path.stem.lower()
        # Убираем номера и разделители
        name = re.sub(r'[_\s]*\d+[_\s]*', '', name)
        name = re.sub(r'[_\-\s]+', '', name)
        return name

    # Создаём словарь текстовых файлов по номеру
    text_by_number: Dict[int, Path] = {}
    text_by_name: Dict[str, Path] = {}

    for text_file in text_files:
        num = extract_number(text_file.stem)
        if num is not None:
            text_by_number[num] = text_file

        base = get_base_name(text_file)
        text_by_name[base] = text_file

    # Сопоставляем аудиофайлы с текстовыми
    for audio_file in sorted(audio_files):
        text_file = None

        # Способ 1: по номеру главы
        num = extract_number(audio_file.stem)
        if num is not None and num in text_by_number:
            text_file = text_by_number[num]

        # Способ 2: по базовому имени
        if text_file is None:
            base = get_base_name(audio_file)
            if base in text_by_name:
                text_file = text_by_name[base]

        # Способ 3: точное совпадение stem
        if text_file is None:
            for tf in text_files:
                if tf.stem.lower() == audio_file.stem.lower():
                    text_file = tf
                    break

        if text_file:
            pairs.append((audio_file, text_file))
            print(f"    ✓ {audio_file.name} <-> {text_file.name}")
        else:
            print(f"    ✗ {audio_file.name} — текст не найден")

    return pairs


def is_chapter_up_to_date(
    audio_path: Path,
    text_path: Path,
    output_base_dir: Path
) -> bool:
    """
    Проверяет, свежее ли результат (filtered.json) по отношению к исходным файлам.

    Возвращает True если пропускать не нужно (результат устарел или отсутствует).
    Возвращает False если результат свежее обоих исходников.
    """
    chapter_id = audio_path.stem
    filtered_path = output_base_dir / chapter_id / f'{chapter_id}_filtered.json'

    if not filtered_path.exists():
        return False

    filtered_mtime = filtered_path.stat().st_mtime
    audio_mtime = audio_path.stat().st_mtime
    text_mtime = text_path.stat().st_mtime

    return filtered_mtime > audio_mtime and filtered_mtime > text_mtime


def process_chapter(
    audio_path: Path,
    text_path: Path,
    output_base_dir: Path,
    skip_transcribe: bool = False,
    transcript_dir: Optional[Path] = None
) -> ChapterResult:
    """
    Обрабатывает одну главу.

    Args:
        audio_path: путь к аудиофайлу
        text_path: путь к текстовому файлу
        output_base_dir: базовая папка для результатов
        skip_transcribe: пропустить транскрибацию
        transcript_dir: папка с готовыми транскрипциями
    """
    from pipeline import run_pipeline, TranscriptValidationError

    result = ChapterResult(
        chapter_name=audio_path.stem,
        audio_file=str(audio_path),
        text_file=str(text_path),
        started_at=datetime.now().isoformat()
    )

    try:
        result.status = 'processing'

        # Определяем папку результатов для этой главы
        output_dir = output_base_dir / audio_path.stem
        result.output_dir = str(output_dir)

        # Ищем готовую транскрипцию
        transcript_path = None
        if skip_transcribe and transcript_dir:
            # Используем FileNaming для поиска по конвенции
            if HAS_CONFIG:
                chapter_id = FileNaming.get_chapter_id(audio_path)
                standard_name = FileNaming.build_filename(chapter_id, 'transcript')
                standard_path = transcript_dir / standard_name
                if standard_path.exists():
                    transcript_path = str(standard_path)

            # Fallback: ищем по имени файла
            if transcript_path is None:
                possible_names = [
                    f"{audio_path.stem}_transcript.json",
                    f"{audio_path.stem}.json",
                    audio_path.stem + '_yandex_transcript.json',
                ]
                for name in possible_names:
                    path = transcript_dir / name
                    if path.exists():
                        transcript_path = str(path)
                        break

        # Запускаем пайплайн
        start_time = datetime.now()

        pipeline_results = run_pipeline(
            str(audio_path),
            str(text_path),
            output_dir=str(output_dir),
            skip_transcribe=skip_transcribe,
            transcript_path=transcript_path,
            web=False
        )

        end_time = datetime.now()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Читаем статистику из финального отчёта
        if pipeline_results.get('filtered'):
            report_path = pipeline_results['filtered']
            result.report_file = report_path

            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)

            result.real_errors = len(report.get('errors', []))
            result.errors_filtered = report.get('filtered_count', 0)
            result.errors_found = result.real_errors + result.errors_filtered

        if pipeline_results.get('docx'):
            result.docx_file = pipeline_results['docx']

        result.status = 'completed'

    except TranscriptValidationError as e:
        result.status = 'failed'
        result.error_message = f"Ошибка валидации транскрипции: {e}"

    except FileNotFoundError as e:
        result.status = 'failed'
        result.error_message = f"Файл не найден: {e}"

    except Exception as e:
        result.status = 'failed'
        result.error_message = str(e)

    result.completed_at = datetime.now().isoformat()
    return result


def run_batch(
    audio_dir: str,
    text_dir: str,
    output_dir: Optional[str] = None,
    audio_pattern: str = '*.mp3',
    text_pattern: str = '*.docx',
    skip_transcribe: bool = False,
    transcript_dir: Optional[str] = None,
    max_chapters: Optional[int] = None,
    continue_on_error: bool = True,
    force: bool = False,
    parallel: bool = False,
    max_workers: int = 3,
    skip_up_to_date: bool = False
) -> BatchResult:
    """
    Запускает пакетную обработку глав.

    Args:
        audio_dir: папка с аудиофайлами
        text_dir: папка с текстовыми файлами
        output_dir: папка для результатов
        audio_pattern: glob-паттерн для аудио
        text_pattern: glob-паттерн для текстов
        skip_transcribe: пропустить транскрибацию
        transcript_dir: папка с готовыми транскрипциями
        max_chapters: максимальное число глав (для тестирования)
        continue_on_error: продолжать при ошибках
        force: перезаписывать существующие результаты
        parallel: использовать параллельную обработку глав
        max_workers: максимальное число параллельных процессов (по умолчанию 3)
        skip_up_to_date: пропускать главы, у которых результат свежее исходников
    """
    result = BatchResult(started_at=datetime.now().isoformat())

    print(f"\n{'#'*60}")
    print(f"  ПАКЕТНАЯ ОБРАБОТКА ГЛАВ")
    print(f"{'#'*60}")
    print(f"  Аудио: {audio_dir}")
    print(f"  Тексты: {text_dir}")
    print(f"{'#'*60}\n")

    # Находим пары файлов
    pairs = find_chapter_pairs(audio_dir, text_dir, audio_pattern, text_pattern)

    if not pairs:
        print("\n✗ Не найдено ни одной пары аудио+текст")
        return result

    if max_chapters:
        pairs = pairs[:max_chapters]

    print(f"\n  Найдено пар: {len(pairs)}")

    # Определяем папку для результатов
    if output_dir:
        output_base = Path(output_dir)
    else:
        # Используем RESULTS_DIR из config.py
        batch_name = 'batch_' + datetime.now().strftime('%Y%m%d_%H%M')
        output_base = RESULTS_DIR / batch_name

    # Проверяем существование папки
    if output_base.exists() and not force:
        action = 'overwrite' if force else 'ask'
        if not check_file_exists(output_base, action=action):
            print(f"  ⚠ Папка уже существует: {output_base}")
            print(f"  Используйте --force для перезаписи")

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"  Результаты: {output_base}\n")

    # Фильтрация up-to-date глав
    if skip_up_to_date and not force:
        filtered_pairs = []
        for audio_path, text_path in pairs:
            if is_chapter_up_to_date(audio_path, text_path, output_base):
                print(f"    ⏭ {audio_path.stem} — результат актуален, пропуск")
                result.skipped += 1
                result.chapters.append(ChapterResult(
                    chapter_name=audio_path.stem,
                    audio_file=str(audio_path),
                    text_file=str(text_path),
                    status='skipped'
                ))
            else:
                filtered_pairs.append((audio_path, text_path))
        pairs = filtered_pairs
        if result.skipped:
            print(f"  Пропущено актуальных: {result.skipped}")

    result.total_chapters = len(pairs) + result.skipped

    # Обрабатываем главы
    transcript_dir_path = Path(transcript_dir) if transcript_dir else None

    if parallel and len(pairs) > 1:
        # Параллельная обработка через ProcessPoolExecutor
        print(f"  Режим: параллельный ({min(max_workers, len(pairs))} процессов)")
        futures = {}
        with ProcessPoolExecutor(max_workers=min(max_workers, len(pairs))) as executor:
            for audio_path, text_path in pairs:
                future = executor.submit(
                    process_chapter,
                    audio_path,
                    text_path,
                    output_base,
                    skip_transcribe=skip_transcribe,
                    transcript_dir=transcript_dir_path
                )
                futures[future] = audio_path.stem

            for future in as_completed(futures):
                chapter_name = futures[future]
                try:
                    chapter_result = future.result()
                except Exception as e:
                    chapter_result = ChapterResult(
                        chapter_name=chapter_name,
                        audio_file='',
                        text_file='',
                        status='failed',
                        error_message=str(e)
                    )

                result.chapters.append(chapter_result)

                if chapter_result.status == 'completed':
                    result.completed += 1
                    result.total_errors_found += chapter_result.errors_found
                    result.total_real_errors += chapter_result.real_errors
                    result.total_duration_seconds += chapter_result.duration_seconds
                    print(f"  ✓ {chapter_name}: {chapter_result.real_errors} ошибок")
                else:
                    result.failed += 1
                    print(f"  ✗ {chapter_name}: {chapter_result.error_message}")

                    if not continue_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        print("\n✗ Прерывание из-за ошибки")
                        break
    else:
        # Последовательная обработка (по умолчанию)
        iterator = pairs
        if HAS_TQDM:
            iterator = tqdm(pairs, desc="Обработка глав", unit="глава")

        for audio_path, text_path in iterator:
            chapter_name = audio_path.stem

            if not HAS_TQDM:
                print(f"\n{'='*60}")
                print(f"  Глава: {chapter_name}")
                print(f"{'='*60}")

            chapter_result = process_chapter(
                audio_path,
                text_path,
                output_base,
                skip_transcribe=skip_transcribe,
                transcript_dir=transcript_dir_path
            )

            result.chapters.append(chapter_result)

            if chapter_result.status == 'completed':
                result.completed += 1
                result.total_errors_found += chapter_result.errors_found
                result.total_real_errors += chapter_result.real_errors
                result.total_duration_seconds += chapter_result.duration_seconds

                if not HAS_TQDM:
                    print(f"  ✓ Завершено: {chapter_result.real_errors} ошибок")
            else:
                result.failed += 1
                if not HAS_TQDM:
                    print(f"  ✗ Ошибка: {chapter_result.error_message}")

                if not continue_on_error:
                    print("\n✗ Прерывание из-за ошибки")
                    break

    result.completed_at = datetime.now().isoformat()

    # Сохраняем общий отчёт
    report_path = output_base / 'batch_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    # Выводим итоги
    print(f"\n{'#'*60}")
    print(f"  ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ")
    print(f"{'#'*60}")
    print(f"  Всего глав: {result.total_chapters}")
    print(f"  Обработано успешно: {result.completed}")
    print(f"  Ошибки: {result.failed}")
    print(f"  Найдено ошибок чтеца: {result.total_real_errors}")
    print(f"  Отфильтровано ложных: {result.total_errors_found - result.total_real_errors}")
    print(f"  Общее время: {format_duration(result.total_duration_seconds)}")
    print(f"  Отчёт: {report_path}")
    print(f"{'#'*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Пакетная обработка глав аудиокниги',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python batch_pipeline.py --audio-dir Главы/аудио --text-dir Главы/тексты
  python batch_pipeline.py -a . -t . --pattern "глава*.mp3"
  python batch_pipeline.py -a . -t . --skip-transcribe --transcript-dir Транскрипции
        """
    )
    parser.add_argument('--audio-dir', '-a', required=True,
                        help='Папка с аудиофайлами')
    parser.add_argument('--text-dir', '-t', required=True,
                        help='Папка с текстовыми файлами')
    parser.add_argument('--output-dir', '-o',
                        help='Папка для результатов')
    parser.add_argument('--audio-pattern', default='*.mp3',
                        help='Glob-паттерн для аудио (по умолчанию: *.mp3)')
    parser.add_argument('--text-pattern', default='*.docx',
                        help='Glob-паттерн для текстов (по умолчанию: *.docx)')
    parser.add_argument('--skip-transcribe', '-s', action='store_true',
                        help='Пропустить транскрибацию (использовать готовые JSON)')
    parser.add_argument('--transcript-dir',
                        help='Папка с готовыми транскрипциями')
    parser.add_argument('--max-chapters', '-m', type=int,
                        help='Максимальное число глав (для тестирования)')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Остановиться при первой ошибке')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Перезаписать существующие результаты')
    parser.add_argument('--parallel', '-P', action='store_true',
                        help='Параллельная обработка глав')
    parser.add_argument('--workers', '-w', type=int, default=3,
                        help='Число параллельных процессов (по умолчанию: 3)')
    parser.add_argument('--only-new', action='store_true',
                        help='Пропускать главы, у которых результат свежее исходников')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  Batch Pipeline v{VERSION}")
    print(f"{'#'*60}")

    try:
        run_batch(
            audio_dir=args.audio_dir,
            text_dir=args.text_dir,
            output_dir=args.output_dir,
            audio_pattern=args.audio_pattern,
            text_pattern=args.text_pattern,
            skip_transcribe=args.skip_transcribe,
            transcript_dir=args.transcript_dir,
            max_chapters=args.max_chapters,
            continue_on_error=not args.stop_on_error,
            force=args.force,
            parallel=args.parallel,
            max_workers=args.workers,
            skip_up_to_date=args.only_new
        )
    except KeyboardInterrupt:
        print("\n\n✗ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
