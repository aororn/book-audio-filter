#!/usr/bin/env python3
"""
Database Schema v2.0 — Расширенная схема БД с контекстами

Новые поля:
- Временные окна: window_start, window_end, context_start, context_end
- Позиции в файлах: pos_transcript, pos_original, pos_normalized
- Контексты из 4 источников: context_transcript, context_normalized, context_original, context_aligned
- Связи между ошибками: linked_errors, link_type, merged_form, split_parts
- Сегменты выравнивания: segment_id, is_boundary

Новые таблицы:
- error_links — связи между ошибками (для merge/split артефактов)
- alignment_segments — сегменты выравнивания

Использование:
    python db_schema_v2.py migrate         # Миграция существующей БД
    python db_schema_v2.py create --new    # Создать новую БД с v2 схемой
    python db_schema_v2.py info            # Информация о схеме

v2.0 (2026-01-31): Начальная версия
"""

VERSION = '2.2.0'  # v2.2: Унифицированный путь к БД из config.py
VERSION_DATE = '2026-01-31'

import argparse
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Путь к БД — ЕДИНЫЙ источник из config.py
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

try:
    from config import FALSE_POSITIVES_DB, DICTIONARIES_DIR
    DB_PATH = FALSE_POSITIVES_DB
    DB_BACKUP_PATH = DICTIONARIES_DIR / 'false_positives_v1_backup.db'
except ImportError:
    DB_PATH = PROJECT_DIR / 'Словари' / 'false_positives.db'
    DB_BACKUP_PATH = PROJECT_DIR / 'Словари' / 'false_positives_v1_backup.db'


# =============================================================================
# СХЕМА V2
# =============================================================================

SCHEMA_V2_ERRORS = '''
-- Расширенная таблица errors v2.0
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- ===== ИДЕНТИФИКАЦИЯ =====
    error_id TEXT,                          -- UUID ошибки (NEW v2.0)

    -- ===== БАЗОВЫЕ ПОЛЯ =====
    wrong TEXT NOT NULL,                    -- Слово из транскрипции (распознано)
    correct TEXT NOT NULL,                  -- Слово из оригинала (книга)
    error_type TEXT NOT NULL,               -- substitution, insertion, deletion
    chapter INTEGER NOT NULL,               -- 1-5

    -- ===== ВРЕМЕННЫЕ МЕТКИ =====
    time_seconds REAL,                      -- Точное время ошибки
    time_end_seconds REAL,                  -- Конец слова (NEW v2.0)
    time_label TEXT,                        -- Формат "MM:SS"
    window_start REAL,                      -- Начало окна выравнивания (NEW v2.0)
    window_end REAL,                        -- Конец окна выравнивания (NEW v2.0)
    context_start REAL,                     -- Начало расширенного контекста (NEW v2.0)
    context_end REAL,                       -- Конец расширенного контекста (NEW v2.0)

    -- ===== ПОЗИЦИИ В ФАЙЛАХ (NEW v2.0) =====
    pos_transcript INTEGER,                 -- Индекс слова в транскрипции
    pos_transcript_char INTEGER,            -- Позиция символа в транскрипции TXT
    pos_normalized INTEGER,                 -- Индекс в нормализованном файле
    pos_original INTEGER,                   -- Индекс слова в оригинале
    pos_original_char INTEGER,              -- Позиция символа в оригинале TXT

    -- ===== КОНТЕКСТЫ (расширенные в v2.0) =====
    context TEXT,                           -- Контекст из оригинала (старое поле)
    transcript_context TEXT,                -- Контекст из транскрипции (старое поле)
    context_transcript TEXT,                -- JSON массив слов из транскрипции (NEW v2.0)
    context_normalized TEXT,                -- JSON массив слов из нормализованного (NEW v2.0)
    context_original TEXT,                  -- JSON массив слов из оригинала (NEW v2.0)
    context_aligned TEXT,                   -- Весь сегмент выравнивания (NEW v2.0)

    -- ===== СВЯЗАННЫЕ ОШИБКИ (NEW v2.0) =====
    linked_errors TEXT,                     -- JSON массив ID связанных ошибок
    link_type TEXT,                         -- merge_artifact, split_artifact, adjacent
    merged_form TEXT,                       -- "навстречу" (если это split artifact)
    split_parts TEXT,                       -- JSON ["на", "встречу"] (если это merge artifact)

    -- ===== СЕГМЕНТЫ (NEW v2.0) =====
    segment_id INTEGER DEFAULT -1,          -- ID сегмента выравнивания
    is_boundary INTEGER DEFAULT 0,          -- На границе сегмента?

    -- ===== GOLDEN СТАТУС =====
    is_golden INTEGER DEFAULT 0,            -- 1 если реальная ошибка чтеца

    -- ===== РЕЗУЛЬТАТ ФИЛЬТРАЦИИ =====
    is_filtered INTEGER DEFAULT 0,          -- 1 если отфильтровано как FP
    filter_reason TEXT,                     -- Название фильтра

    -- ===== МОРФОЛОГИЯ =====
    lemma_wrong TEXT,
    lemma_correct TEXT,
    pos_wrong TEXT,
    pos_correct TEXT,
    same_lemma INTEGER DEFAULT 0,           -- 1 если одинаковая лемма
    same_pos INTEGER DEFAULT 0,             -- 1 если одинаковая часть речи

    -- ===== СЕМАНТИКА =====
    semantic_similarity REAL DEFAULT 0,     -- Косинусное сходство 0-1

    -- ===== ЧАСТОТНОСТЬ =====
    frequency_wrong REAL DEFAULT 0,
    frequency_correct REAL DEFAULT 0,

    -- ===== ФОНЕТИКА =====
    phonetic_similarity REAL DEFAULT 0,     -- Из compared.json
    levenshtein INTEGER DEFAULT 0,          -- Расстояние Левенштейна

    -- ===== МЕТАДАННЫЕ =====
    created_at TEXT NOT NULL,
    updated_at TEXT,                        -- NEW v2.0
    schema_version INTEGER DEFAULT 2,       -- NEW v2.0

    -- Уникальность по главе + время + слова
    UNIQUE(chapter, time_seconds, wrong, correct)
);
'''

SCHEMA_V2_ERROR_LINKS = '''
-- Таблица связей между ошибками (NEW v2.0)
CREATE TABLE IF NOT EXISTS error_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    link_id TEXT NOT NULL,                  -- UUID связи

    error1_id TEXT NOT NULL,                -- ID первой ошибки
    error2_id TEXT NOT NULL,                -- ID второй ошибки

    link_type TEXT NOT NULL,                -- merge_artifact, split_artifact, adjacent
    pattern TEXT,                           -- "на+встречу=навстречу"
    original_parts TEXT,                    -- JSON ["на", "встречу"]
    merged_form TEXT,                       -- "навстречу"

    confidence REAL DEFAULT 1.0,            -- Уверенность в связи (0-1)

    chapter INTEGER,                        -- Глава
    time_start REAL,                        -- Время начала
    time_end REAL,                          -- Время конца

    created_at TEXT NOT NULL,

    UNIQUE(error1_id, error2_id)
);
'''

SCHEMA_V2_ALIGNMENT_SEGMENTS = '''
-- Таблица сегментов выравнивания (NEW v2.0)
CREATE TABLE IF NOT EXISTS alignment_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    chapter INTEGER NOT NULL,
    segment_idx INTEGER NOT NULL,           -- Индекс сегмента в главе

    anchor_before TEXT,                     -- Якорь до сегмента
    anchor_after TEXT,                      -- Якорь после сегмента

    time_start REAL,
    time_end REAL,

    original_start INTEGER,                 -- Начало в оригинале (индекс слова)
    original_end INTEGER,                   -- Конец в оригинале
    transcript_start INTEGER,               -- Начало в транскрипции
    transcript_end INTEGER,                 -- Конец в транскрипции

    original_text TEXT,                     -- Текст сегмента из оригинала
    transcript_text TEXT,                   -- Текст сегмента из транскрипции

    errors_count INTEGER DEFAULT 0,         -- Количество ошибок в сегменте
    error_ids TEXT,                         -- JSON массив ID ошибок

    created_at TEXT NOT NULL,

    UNIQUE(chapter, segment_idx)
);
'''

SCHEMA_V2_ERROR_HISTORY = '''
-- Таблица истории изменений ошибок (NEW v2.1)
-- Отслеживает все изменения статуса ошибок при каждом прогоне
CREATE TABLE IF NOT EXISTS error_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Идентификация прогона
    run_id TEXT NOT NULL,                   -- UUID прогона (группирует изменения)
    run_timestamp TEXT NOT NULL,            -- Время прогона
    project_version TEXT,                   -- Версия проекта (14.6.0)
    filter_version TEXT,                    -- Версия engine.py (9.11.0)

    -- Идентификация ошибки
    error_id TEXT NOT NULL,                 -- UUID ошибки
    chapter INTEGER NOT NULL,
    time_seconds REAL,
    wrong TEXT,
    correct TEXT,
    error_type TEXT,

    -- Что произошло
    action TEXT NOT NULL,                   -- created, filtered, unfiltered, deleted, golden_added, golden_removed

    -- Детали изменения
    old_is_filtered INTEGER,                -- Предыдущий статус (NULL для created)
    new_is_filtered INTEGER,                -- Новый статус
    old_filter_reason TEXT,                 -- Предыдущая причина
    new_filter_reason TEXT,                 -- Новая причина
    old_is_golden INTEGER,
    new_is_golden INTEGER,

    -- Контекст
    context TEXT,                           -- Контекст ошибки для понимания

    created_at TEXT NOT NULL
);
'''

SCHEMA_V2_SYNC_RUNS = '''
-- Таблица прогонов синхронизации (NEW v2.1)
-- Метаданные каждого прогона populate_db
CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL UNIQUE,            -- UUID прогона
    run_timestamp TEXT NOT NULL,            -- Время начала
    run_finished TEXT,                      -- Время завершения

    -- Версии
    project_version TEXT,
    filter_version TEXT,
    smart_compare_version TEXT,

    -- Статистика
    total_errors_before INTEGER,            -- Ошибок до прогона
    total_errors_after INTEGER,             -- Ошибок после
    errors_created INTEGER DEFAULT 0,       -- Новых ошибок
    errors_deleted INTEGER DEFAULT 0,       -- Удалённых
    errors_filtered INTEGER DEFAULT 0,      -- Перешли в filtered
    errors_unfiltered INTEGER DEFAULT 0,    -- Перешли из filtered
    golden_added INTEGER DEFAULT 0,         -- Добавлено в golden
    golden_removed INTEGER DEFAULT 0,       -- Убрано из golden

    -- Флаги
    is_reset INTEGER DEFAULT 0,             -- Был ли это --reset
    chapters_processed TEXT,                -- JSON список глав

    -- Примечание
    note TEXT                               -- Комментарий к прогону
);
'''

SCHEMA_V2_INDEXES = '''
-- Индексы для v2.0
CREATE INDEX IF NOT EXISTS idx_errors_error_id ON errors(error_id);
CREATE INDEX IF NOT EXISTS idx_errors_golden ON errors(is_golden);
CREATE INDEX IF NOT EXISTS idx_errors_filtered ON errors(is_filtered);
CREATE INDEX IF NOT EXISTS idx_errors_chapter ON errors(chapter);
CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type);
CREATE INDEX IF NOT EXISTS idx_errors_same_lemma ON errors(same_lemma);
CREATE INDEX IF NOT EXISTS idx_errors_semantic ON errors(semantic_similarity);
CREATE INDEX IF NOT EXISTS idx_errors_segment_id ON errors(segment_id);
CREATE INDEX IF NOT EXISTS idx_errors_link_type ON errors(link_type);
CREATE INDEX IF NOT EXISTS idx_errors_time ON errors(time_seconds);

CREATE INDEX IF NOT EXISTS idx_error_links_type ON error_links(link_type);
CREATE INDEX IF NOT EXISTS idx_error_links_chapter ON error_links(chapter);

CREATE INDEX IF NOT EXISTS idx_segments_chapter ON alignment_segments(chapter);

-- Индексы для истории (v2.1)
CREATE INDEX IF NOT EXISTS idx_history_run_id ON error_history(run_id);
CREATE INDEX IF NOT EXISTS idx_history_error_id ON error_history(error_id);
CREATE INDEX IF NOT EXISTS idx_history_action ON error_history(action);
CREATE INDEX IF NOT EXISTS idx_history_chapter ON error_history(chapter);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON error_history(run_timestamp);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON sync_runs(run_timestamp);
'''


# =============================================================================
# МИГРАЦИИ
# =============================================================================

def get_schema_version(conn: sqlite3.Connection) -> float:
    """Определяет версию схемы БД (1.0, 2.0, 2.1)"""
    try:
        # Проверяем наличие таблиц v2.1
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        # v2.1: есть error_history и sync_runs
        if 'error_history' in tables and 'sync_runs' in tables:
            return 2.1

        # Проверяем наличие полей v2.0
        cur = conn.execute("PRAGMA table_info(errors)")
        columns = {row[1] for row in cur.fetchall()}

        if 'error_id' in columns and 'linked_errors' in columns:
            return 2.0
        return 1.0
    except Exception:
        return 0


def backup_database(db_path: Path, backup_path: Path) -> bool:
    """Создаёт резервную копию БД"""
    import shutil
    try:
        shutil.copy2(db_path, backup_path)
        print(f"[OK] Бэкап создан: {backup_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка бэкапа: {e}")
        return False


def migrate_v1_to_v2(conn: sqlite3.Connection) -> bool:
    """Миграция с v1 на v2"""
    print("\n[MIGRATE] v1 → v2")

    # Список новых колонок для таблицы errors
    new_columns = [
        ('error_id', 'TEXT'),
        ('time_end_seconds', 'REAL'),
        ('window_start', 'REAL'),
        ('window_end', 'REAL'),
        ('context_start', 'REAL'),
        ('context_end', 'REAL'),
        ('pos_transcript', 'INTEGER'),
        ('pos_transcript_char', 'INTEGER'),
        ('pos_normalized', 'INTEGER'),
        ('pos_original', 'INTEGER'),
        ('pos_original_char', 'INTEGER'),
        ('transcript_context', 'TEXT'),
        ('context_transcript', 'TEXT'),
        ('context_normalized', 'TEXT'),
        ('context_original', 'TEXT'),
        ('context_aligned', 'TEXT'),
        ('linked_errors', 'TEXT'),
        ('link_type', 'TEXT'),
        ('merged_form', 'TEXT'),
        ('split_parts', 'TEXT'),
        ('segment_id', 'INTEGER DEFAULT -1'),
        ('is_boundary', 'INTEGER DEFAULT 0'),
        ('updated_at', 'TEXT'),
        ('schema_version', 'INTEGER DEFAULT 2'),
    ]

    # Получаем существующие колонки
    cur = conn.execute("PRAGMA table_info(errors)")
    existing_columns = {row[1] for row in cur.fetchall()}

    # Добавляем недостающие колонки
    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            try:
                conn.execute(f"ALTER TABLE errors ADD COLUMN {col_name} {col_type}")
                print(f"  + {col_name} ({col_type})")
            except sqlite3.OperationalError as e:
                print(f"  [SKIP] {col_name}: {e}")

    # Создаём новые таблицы
    print("\n[CREATE] Новые таблицы...")
    conn.executescript(SCHEMA_V2_ERROR_LINKS)
    print("  + error_links")
    conn.executescript(SCHEMA_V2_ALIGNMENT_SEGMENTS)
    print("  + alignment_segments")
    conn.executescript(SCHEMA_V2_ERROR_HISTORY)
    print("  + error_history")
    conn.executescript(SCHEMA_V2_SYNC_RUNS)
    print("  + sync_runs")

    # Создаём индексы
    print("\n[CREATE] Индексы...")
    conn.executescript(SCHEMA_V2_INDEXES)
    print("  + все индексы")

    conn.commit()
    return True


def migrate_v2_to_v21(conn: sqlite3.Connection) -> bool:
    """Миграция с v2.0 на v2.1 — добавление таблиц истории"""
    print("\n[MIGRATE] v2.0 → v2.1")

    # Создаём таблицы истории
    print("\n[CREATE] Таблицы истории...")
    conn.executescript(SCHEMA_V2_ERROR_HISTORY)
    print("  + error_history")
    conn.executescript(SCHEMA_V2_SYNC_RUNS)
    print("  + sync_runs")

    # Создаём индексы для истории
    print("\n[CREATE] Индексы истории...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_run_id ON error_history(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_error_id ON error_history(error_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_action ON error_history(action)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_chapter ON error_history(chapter)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON error_history(run_timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON sync_runs(run_timestamp)")
    print("  + все индексы истории")

    conn.commit()
    print("\n[OK] Миграция v2.0 → v2.1 завершена!")
    return True


def create_fresh_v2(db_path: Path) -> bool:
    """Создаёт новую БД с v2 схемой"""
    if db_path.exists():
        print(f"[ERROR] БД уже существует: {db_path}")
        print("  Используйте --force для перезаписи")
        return False

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    print("\n[CREATE] Схема v2.1...")
    conn.executescript(SCHEMA_V2_ERRORS)
    print("  + errors")
    conn.executescript(SCHEMA_V2_ERROR_LINKS)
    print("  + error_links")
    conn.executescript(SCHEMA_V2_ALIGNMENT_SEGMENTS)
    print("  + alignment_segments")
    conn.executescript(SCHEMA_V2_ERROR_HISTORY)
    print("  + error_history")
    conn.executescript(SCHEMA_V2_SYNC_RUNS)
    print("  + sync_runs")
    conn.executescript(SCHEMA_V2_INDEXES)
    print("  + индексы")

    conn.commit()
    conn.close()

    print(f"\n[OK] БД создана: {db_path}")
    return True


def show_schema_info(db_path: Path):
    """Показывает информацию о схеме"""
    if not db_path.exists():
        print(f"[ERROR] БД не найдена: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    version = get_schema_version(conn)

    print(f"\n{'='*60}")
    print(f"ИНФОРМАЦИЯ О СХЕМЕ БД")
    print(f"{'='*60}")
    print(f"Путь: {db_path}")
    print(f"Версия схемы: v{version}.0")

    # Таблицы
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    print(f"\nТаблицы ({len(tables)}):")
    for table in tables:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  - {table}: {count} записей")

    # Колонки errors
    print(f"\nКолонки errors:")
    cur = conn.execute("PRAGMA table_info(errors)")
    for row in cur.fetchall():
        col_id, name, dtype, notnull, default, pk = row
        print(f"  {col_id:2d}. {name}: {dtype}")

    # Индексы
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
    indexes = [row[0] for row in cur.fetchall()]
    print(f"\nИндексы ({len(indexes)}):")
    for idx in indexes:
        print(f"  - {idx}")

    conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Database Schema v2.0 — Миграции и управление схемой'
    )

    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # Команда migrate
    migrate_parser = subparsers.add_parser('migrate', help='Миграция существующей БД')
    migrate_parser.add_argument('--no-backup', action='store_true', help='Без бэкапа')

    # Команда create
    create_parser = subparsers.add_parser('create', help='Создать новую БД')
    create_parser.add_argument('--new', action='store_true', help='Новая v2 БД')
    create_parser.add_argument('--force', action='store_true', help='Перезаписать существующую')

    # Команда info
    info_parser = subparsers.add_parser('info', help='Информация о схеме')

    args = parser.parse_args()

    print(f"Database Schema v{VERSION}")
    print("=" * 60)

    if args.command == 'migrate':
        if not DB_PATH.exists():
            print(f"[ERROR] БД не найдена: {DB_PATH}")
            return

        # Бэкап
        if not args.no_backup:
            backup_database(DB_PATH, DB_BACKUP_PATH)

        # Миграция
        conn = sqlite3.connect(str(DB_PATH))
        current_version = get_schema_version(conn)
        print(f"\n[INFO] Текущая версия схемы: v{current_version}")

        if current_version >= 2.1:
            print("[OK] БД уже на версии v2.1 (с историей)")
        elif current_version >= 2.0:
            print("[INFO] БД на версии v2.0, требуется миграция на v2.1")
            migrate_v2_to_v21(conn)
        elif current_version >= 1.0:
            print("[INFO] БД на версии v1.0, требуется миграция на v2.1")
            migrate_v1_to_v2(conn)
            migrate_v2_to_v21(conn)
        else:
            print("[ERROR] Неизвестная версия схемы")

        conn.close()
        show_schema_info(DB_PATH)

    elif args.command == 'create':
        if args.new:
            if args.force and DB_PATH.exists():
                DB_PATH.unlink()
            create_fresh_v2(DB_PATH)
            show_schema_info(DB_PATH)
        else:
            print("Укажите --new для создания новой v2 БД")

    elif args.command == 'info':
        show_schema_info(DB_PATH)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
