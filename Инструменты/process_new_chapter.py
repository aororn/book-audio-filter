#!/usr/bin/env python3
"""
process_new_chapter.py — Скрипт автоматизации обработки новых глав

Полный цикл обработки:
1. Транскрипция аудио через Яндекс SpeechKit
2. Выравнивание с эталонным текстом
3. Генерация отчёта об ошибках
4. Фильтрация ложных срабатываний
5. Добавление новых ложных срабатываний в базу
6. Генерация статистики

Использование:
    python3 process_new_chapter.py <audio_file> <reference_text> [options]

Примеры:
    python3 process_new_chapter.py audio/ch7.mp3 text/chapter7.txt
    python3 process_new_chapter.py audio/ch7.mp3 text/chapter7.txt --chapter 7
    python3 process_new_chapter.py audio/ch7.mp3 text/chapter7.txt --dry-run
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Добавляем путь к модулям
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from false_positives_db import FalsePositivesDB
    HAS_FP_DB = True
except ImportError:
    HAS_FP_DB = False


def log(msg: str, level: str = "INFO"):
    """Логирование с временной меткой."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Запуск команды с логированием."""
    log(f"{description}...")
    log(f"  Команда: {' '.join(cmd)}", "DEBUG")

    if dry_run:
        log("  (dry-run: команда не выполнена)", "DEBUG")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR)
        )
        if result.returncode != 0:
            log(f"  Ошибка: {result.stderr}", "ERROR")
            return None
        return result
    except Exception as e:
        log(f"  Исключение: {e}", "ERROR")
        return None


def check_prerequisites() -> bool:
    """Проверка наличия необходимых файлов и зависимостей."""
    required_files = [
        SCRIPT_DIR / "smart_compare.py",
        SCRIPT_DIR / "filters" / "engine.py",
    ]

    missing = []
    for f in required_files:
        if not f.exists():
            missing.append(str(f))

    if missing:
        log(f"Отсутствуют файлы: {', '.join(missing)}", "ERROR")
        return False

    return True


def transcribe_audio(audio_path: Path, output_dir: Path, dry_run: bool = False) -> Optional[Path]:
    """
    Транскрипция аудио через Яндекс SpeechKit.

    Возвращает путь к JSON-файлу с транскрипцией.
    """
    transcript_path = output_dir / f"{audio_path.stem}_transcript.json"

    # Проверяем, есть ли уже транскрипция
    if transcript_path.exists() and not dry_run:
        log(f"  Транскрипция уже существует: {transcript_path}")
        return transcript_path

    # Команда для транскрипции
    cmd = [
        "python3",
        str(SCRIPT_DIR / "transcribe.py"),
        str(audio_path),
        "-o", str(transcript_path)
    ]

    result = run_command(cmd, f"Транскрипция {audio_path.name}", dry_run)

    if dry_run:
        return transcript_path

    if result and transcript_path.exists():
        return transcript_path

    return None


def align_texts(
    transcript_path: Path,
    reference_path: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Выравнивание транскрипции с эталоном.

    Возвращает путь к JSON-файлу с выравниванием.
    """
    alignment_path = output_dir / f"{reference_path.stem}_alignment.json"

    cmd = [
        "python3",
        str(SCRIPT_DIR / "smart_compare.py"),
        str(transcript_path),
        str(reference_path),
        "-o", str(alignment_path),
        "--format", "json"
    ]

    result = run_command(cmd, "Выравнивание текстов", dry_run)

    if dry_run:
        return alignment_path

    if result and alignment_path.exists():
        return alignment_path

    return None


def generate_report(
    alignment_path: Path,
    output_dir: Path,
    chapter_num: Optional[int] = None,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Генерация отчёта об ошибках.

    Возвращает путь к HTML-отчёту.
    """
    chapter_suffix = f"_ch{chapter_num}" if chapter_num else ""
    report_path = output_dir / f"report{chapter_suffix}.html"

    cmd = [
        "python3",
        str(SCRIPT_DIR / "web_viewer.py"),
        str(alignment_path),
        "-o", str(report_path)
    ]

    result = run_command(cmd, "Генерация HTML-отчёта", dry_run)

    if dry_run:
        return report_path

    if result and report_path.exists():
        return report_path

    return None


def collect_false_positives(
    alignment_path: Path,
    chapter_num: Optional[int] = None,
    source_name: str = "auto",
    dry_run: bool = False
) -> int:
    """
    Сбор ложных срабатываний из отфильтрованных ошибок.

    Возвращает количество добавленных паттернов.
    """
    if not HAS_FP_DB:
        log("Модуль false_positives_db не доступен", "WARNING")
        return 0

    if dry_run:
        log("  (dry-run: ложные срабатывания не собраны)")
        return 0

    try:
        with open(alignment_path) as f:
            data = json.load(f)
    except Exception as e:
        log(f"Ошибка чтения {alignment_path}: {e}", "ERROR")
        return 0

    # Извлекаем отфильтрованные ошибки
    filtered_errors = data.get("filtered_errors", [])
    if not filtered_errors:
        log("  Нет отфильтрованных ошибок для добавления")
        return 0

    db = FalsePositivesDB()
    added = 0

    source = f"chapter_{chapter_num}" if chapter_num else source_name

    for error in filtered_errors:
        original = error.get("original", "")
        recognized = error.get("recognized", "")
        error_type = error.get("type", "unknown")

        if original or recognized:
            pattern = f"{original}→{recognized}"
            try:
                db.add_pattern(
                    pattern=pattern,
                    source=source,
                    error_type=error_type
                )
                added += 1
            except Exception:
                pass  # Паттерн уже существует

    log(f"  Добавлено {added} паттернов в базу ложных срабатываний")
    return added


def print_statistics(alignment_path: Path, dry_run: bool = False):
    """Вывод статистики по обработанной главе."""
    if dry_run:
        log("  (dry-run: статистика не доступна)")
        return

    try:
        with open(alignment_path) as f:
            data = json.load(f)
    except Exception as e:
        log(f"Ошибка чтения статистики: {e}", "ERROR")
        return

    errors = data.get("errors", [])
    filtered = data.get("filtered_errors", [])

    total = len(errors) + len(filtered)
    filter_rate = len(filtered) / total * 100 if total > 0 else 0

    log("=" * 50)
    log("СТАТИСТИКА ОБРАБОТКИ")
    log("=" * 50)
    log(f"  Всего найдено различий: {total}")
    log(f"  Отфильтровано: {len(filtered)} ({filter_rate:.1f}%)")
    log(f"  Реальных ошибок: {len(errors)}")

    # Разбивка по типам
    type_counts: Dict[str, int] = {}
    for e in errors:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        log("  По типам:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            log(f"    {t}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Автоматизация обработки новых глав аудиокниги",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Путь к аудиофайлу"
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Путь к эталонному тексту"
    )
    parser.add_argument(
        "-c", "--chapter",
        type=int,
        help="Номер главы (для именования файлов)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Директория для выходных файлов (по умолчанию: рядом с эталоном)"
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Пропустить транскрипцию (использовать существующую)"
    )
    parser.add_argument(
        "--skip-fp",
        action="store_true",
        help="Не добавлять ложные срабатывания в базу"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать что будет сделано, без выполнения"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробный вывод"
    )

    args = parser.parse_args()

    # Проверки
    if not args.audio.exists():
        log(f"Аудиофайл не найден: {args.audio}", "ERROR")
        sys.exit(1)

    if not args.reference.exists():
        log(f"Эталонный текст не найден: {args.reference}", "ERROR")
        sys.exit(1)

    if not check_prerequisites():
        sys.exit(1)

    # Определяем выходную директорию
    output_dir = args.output or args.reference.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("ОБРАБОТКА НОВОЙ ГЛАВЫ")
    log("=" * 60)
    log(f"  Аудио: {args.audio}")
    log(f"  Эталон: {args.reference}")
    log(f"  Выходная директория: {output_dir}")
    if args.chapter:
        log(f"  Номер главы: {args.chapter}")
    if args.dry_run:
        log("  РЕЖИМ: dry-run (тестовый запуск)")
    log("")

    # 1. Транскрипция
    if not args.skip_transcribe:
        transcript_path = transcribe_audio(args.audio, output_dir, args.dry_run)
        if not transcript_path and not args.dry_run:
            log("Ошибка транскрипции", "ERROR")
            sys.exit(1)
    else:
        # Ищем существующую транскрипцию
        transcript_path = output_dir / f"{args.audio.stem}_transcript.json"
        if not transcript_path.exists() and not args.dry_run:
            log(f"Транскрипция не найдена: {transcript_path}", "ERROR")
            sys.exit(1)
        log(f"Используется существующая транскрипция: {transcript_path}")

    # 2. Выравнивание
    alignment_path = align_texts(
        transcript_path, args.reference, output_dir, args.dry_run
    )
    if not alignment_path and not args.dry_run:
        log("Ошибка выравнивания", "ERROR")
        sys.exit(1)

    # 3. Генерация отчёта
    report_path = generate_report(
        alignment_path, output_dir, args.chapter, args.dry_run
    )
    if report_path:
        log(f"  Отчёт: {report_path}")

    # 4. Сбор ложных срабатываний
    if not args.skip_fp:
        collect_false_positives(
            alignment_path, args.chapter, dry_run=args.dry_run
        )

    # 5. Статистика
    print_statistics(alignment_path, args.dry_run)

    log("")
    log("=" * 60)
    log("ОБРАБОТКА ЗАВЕРШЕНА")
    log("=" * 60)

    if report_path and not args.dry_run:
        log(f"Отчёт доступен: {report_path}")


if __name__ == "__main__":
    main()
