#!/usr/bin/env python3
"""
False Positives Database v2.0 - SQLite-based storage for false positives tracking

Замена JSON-хранилища на SQLite для:
- Быстрого поиска и фильтрации
- Сложной аналитики (GROUP BY, ORDER BY)
- Масштабируемости при росте данных
- ML-подготовки (морфологические признаки)

Использование:
    from false_positives_db import FalsePositivesDB

    db = FalsePositivesDB()
    db.add_false_positive("живем", "живы", "substitution", "01_yandex", "контекст...")
    top = db.get_top_patterns(limit=20)
    db.mark_resolved("живем→живы", filter_name="grammar_ending")

CLI:
    python false_positives_db.py stats           # Статистика
    python false_positives_db.py top 20          # Топ-20 паттернов
    python false_positives_db.py migrate         # Миграция из JSON
    python false_positives_db.py export --csv    # Экспорт в CSV
    python false_positives_db.py update-morph    # Обновить морф-признаки
    python false_positives_db.py mark-golden     # Разметить golden standard

Changelog:
    v2.0 (2026-01-26): Расширение для ML
        - Версионирование схемы (SCHEMA_VERSION = 2)
        - Новые колонки: is_golden, lemma1/2, pos1/2, aspect1/2, levenshtein, same_lemma, ml_score
        - Автомиграция при запуске
        - update_morphology() — заполнение морф-признаков
        - mark_golden() — разметка из golden standard
    v1.0 (2026-01-25): Начальная версия
        - SQLite storage с 3 таблицами (patterns, occurrences, filter_rules)
        - Автоклассификация паттернов
        - Миграция из JSON
        - CLI интерфейс
"""

VERSION = '2.0.0'
VERSION_DATE = '2026-01-26'
SCHEMA_VERSION = 2

import sqlite3
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


# =============================================================================
# ИМПОРТ КОНФИГУРАЦИИ
# =============================================================================

try:
    from config import TEMP_DIR, DICTIONARIES_DIR, TESTS_DIR, FALSE_POSITIVES_DB
    DICTS_DIR = DICTIONARIES_DIR  # alias
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    TEMP_DIR = Path(__file__).parent.parent / 'Темп'
    DICTS_DIR = Path(__file__).parent.parent / 'Словари'
    TESTS_DIR = Path(__file__).parent.parent / 'Тесты'
    FALSE_POSITIVES_DB = TEMP_DIR / 'false_positives.db'

# Импорт морфологии для ML-признаков
try:
    from morphology import get_lemma, get_pos, get_word_info, HAS_PYMORPHY
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False
    HAS_PYMORPHY = False


# =============================================================================
# СТРУКТУРЫ ДАННЫХ
# =============================================================================

@dataclass
class Pattern:
    """Паттерн ложного срабатывания"""
    id: int
    wrong: str
    correct: str
    error_type: str  # substitution, insertion, deletion
    pattern_key: str  # "wrong→correct"
    count: int
    category: str  # grammar_ending, phonetic, prefix_variant, short_word, compound_word, character_name, unknown
    status: str  # active, resolved, ignored
    filter_applied: Optional[str]  # какой фильтр закрыл этот паттерн
    first_seen: str
    last_seen: str


@dataclass
class Occurrence:
    """Конкретное вхождение паттерна"""
    id: int
    pattern_id: int
    source: str  # "01_48kbps", "02_yandex"
    time_seconds: Optional[int]
    context: str
    created_at: str


# =============================================================================
# КЛАССИФИКАТОР ПАТТЕРНОВ
# =============================================================================

def classify_pattern(wrong: str, correct: str, error_type: str) -> str:
    """
    Автоматическая классификация паттерна ложного срабатывания.

    Returns:
        Категория: grammar_ending, phonetic, prefix_variant, short_word,
                  compound_word, character_name, alignment_artifact, unknown
    """
    # Нормализация
    w = wrong.lower().replace('ё', 'е').strip()
    c = correct.lower().replace('ё', 'е').strip()

    # 1. Короткие слова (≤2 символа)
    if len(w) <= 2 or len(c) <= 2:
        return "short_word"

    # 2. Insertion/Deletion без пары — артефакт выравнивания
    if error_type == 'insertion' and not c:
        return "alignment_artifact"
    if error_type == 'deletion' and not w:
        return "alignment_artifact"

    # 3. Грамматические окончания (одинаковая основа, разные окончания)
    min_len = min(len(w), len(c))
    if min_len >= 4:
        # Ищем общую основу
        common = 0
        for i in range(min_len):
            if w[i] == c[i]:
                common += 1
            else:
                break

        if common >= 3 and common >= min_len * 0.6:
            return "grammar_ending"

    # 4. Приставки (одно слово = другое + приставка)
    prefixes = ['не', 'на', 'по', 'от', 'у', 'вы', 'за', 'до', 'под', 'при', 'пере', 'с', 'в', 'об']
    for prefix in prefixes:
        if w.startswith(prefix) and w[len(prefix):] == c:
            return "prefix_variant"
        if c.startswith(prefix) and c[len(prefix):] == w:
            return "prefix_variant"

    # 5. Фонетическое сходство (Левенштейн <= 2 для длинных слов)
    if min_len >= 5:
        dist = _levenshtein_distance(w, c)
        if dist <= 2:
            return "phonetic"

    # 6. Составные слова (дефисные)
    if '-' in w or '-' in c:
        return "compound_word"

    return "unknown"


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Простая реализация расстояния Левенштейна"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# =============================================================================
# ОСНОВНОЙ КЛАСС БАЗЫ ДАННЫХ
# =============================================================================

class FalsePositivesDB:
    """
    SQLite-хранилище для ложных срабатываний.

    Схема:
    - patterns: уникальные паттерны (wrong→correct)
    - occurrences: конкретные вхождения с контекстом
    - filter_rules: правила фильтрации и их эффективность
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Инициализация базы данных.

        Args:
            db_path: Путь к файлу БД (по умолчанию из config.py)
        """
        if db_path is None:
            db_path = FALSE_POSITIVES_DB

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate_schema()

    def _create_tables(self):
        """Создание таблиц"""
        self.conn.executescript('''
            -- Паттерны ложных срабатываний
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wrong TEXT NOT NULL,
                correct TEXT NOT NULL,
                error_type TEXT NOT NULL,  -- substitution, insertion, deletion
                pattern_key TEXT UNIQUE NOT NULL,  -- "wrong→correct"
                count INTEGER DEFAULT 1,
                category TEXT DEFAULT 'unknown',  -- классификация
                status TEXT DEFAULT 'active',  -- active, resolved, ignored
                filter_applied TEXT,  -- какой фильтр закрыл
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            );

            -- Индексы для быстрого поиска
            CREATE INDEX IF NOT EXISTS idx_patterns_key ON patterns(pattern_key);
            CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
            CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(category);
            CREATE INDEX IF NOT EXISTS idx_patterns_count ON patterns(count DESC);

            -- Конкретные вхождения паттернов
            CREATE TABLE IF NOT EXISTS occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                source TEXT NOT NULL,  -- "01_48kbps", "02_yandex"
                time_seconds INTEGER,
                context TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES patterns(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_occurrences_pattern ON occurrences(pattern_id);
            CREATE INDEX IF NOT EXISTS idx_occurrences_source ON occurrences(source);

            -- Правила фильтрации
            CREATE TABLE IF NOT EXISTS filter_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,  -- homophone, grammar_ending, etc.
                description TEXT,
                patterns_resolved INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            -- Версия схемы
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
        ''')
        self.conn.commit()

    def _migrate_schema(self):
        """Миграция схемы БД до актуальной версии."""
        # Получаем текущую версию
        cursor = self.conn.execute('SELECT version FROM schema_version LIMIT 1')
        row = cursor.fetchone()
        current_version = row[0] if row else 1

        if current_version >= SCHEMA_VERSION:
            return  # Уже актуальная версия

        # Миграция v1 → v2: добавляем колонки для ML
        if current_version < 2:
            new_columns = [
                ('is_golden', 'INTEGER DEFAULT 0'),
                ('lemma1', 'TEXT'),
                ('lemma2', 'TEXT'),
                ('pos1', 'TEXT'),
                ('pos2', 'TEXT'),
                ('aspect1', 'TEXT'),
                ('aspect2', 'TEXT'),
                ('levenshtein', 'INTEGER'),
                ('same_lemma', 'INTEGER'),
                ('ml_score', 'REAL'),
            ]

            for col_name, col_type in new_columns:
                try:
                    self.conn.execute(f'ALTER TABLE patterns ADD COLUMN {col_name} {col_type}')
                except sqlite3.OperationalError:
                    pass  # Колонка уже существует

            # Создаём индекс для is_golden
            try:
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_patterns_golden ON patterns(is_golden)')
            except sqlite3.OperationalError:
                pass

        # Обновляем версию
        self.conn.execute('DELETE FROM schema_version')
        self.conn.execute('INSERT INTO schema_version (version) VALUES (?)', (SCHEMA_VERSION,))
        self.conn.commit()
        print(f"✓ Схема БД мигрирована до версии {SCHEMA_VERSION}")

    # =========================================================================
    # ДОБАВЛЕНИЕ ДАННЫХ
    # =========================================================================

    def add_false_positive(
        self,
        wrong: str,
        correct: str,
        error_type: str,
        source: str,
        context: str = "",
        time_seconds: Optional[int] = None
    ) -> int:
        """
        Добавляет ложное срабатывание в базу.

        Args:
            wrong: Что услышал Яндекс
            correct: Что должно быть
            error_type: substitution, insertion, deletion
            source: Источник (01_yandex, 02_48kbps)
            context: Контекст из текста
            time_seconds: Время в секундах

        Returns:
            ID паттерна
        """
        pattern_key = f"{wrong}→{correct}"
        now = datetime.now().isoformat()

        # Проверяем существующий паттерн
        cursor = self.conn.execute(
            'SELECT id, count FROM patterns WHERE pattern_key = ?',
            (pattern_key,)
        )
        row = cursor.fetchone()

        if row:
            # Обновляем существующий
            pattern_id = row['id']
            self.conn.execute(
                'UPDATE patterns SET count = count + 1, last_seen = ? WHERE id = ?',
                (now, pattern_id)
            )
        else:
            # Создаём новый
            category = classify_pattern(wrong, correct, error_type)
            cursor = self.conn.execute(
                '''INSERT INTO patterns
                   (wrong, correct, error_type, pattern_key, count, category, status, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, 1, ?, 'active', ?, ?)''',
                (wrong, correct, error_type, pattern_key, category, now, now)
            )
            pattern_id = cursor.lastrowid

        # Добавляем вхождение
        self.conn.execute(
            '''INSERT INTO occurrences (pattern_id, source, time_seconds, context, created_at)
               VALUES (?, ?, ?, ?, ?)''',
            (pattern_id, source, time_seconds, context, now)
        )

        self.conn.commit()
        return pattern_id

    def add_from_report(
        self,
        report_path: Path,
        source: str,
        golden_path: Optional[Path] = None
    ) -> Dict[str, int]:
        """
        Добавляет ложные срабатывания из JSON-отчёта.

        Args:
            report_path: Путь к filtered.json
            source: Идентификатор источника
            golden_path: Путь к golden standard (исключить из добавления)

        Returns:
            Статистика: {'added': N, 'skipped_golden': M}
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # Загружаем golden для исключения
        golden_keys = set()
        if golden_path and golden_path.exists():
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden = json.load(f)
                for err in golden.get('errors', golden if isinstance(golden, list) else []):
                    wrong = err.get('wrong', err.get('spoken', ''))
                    correct = err.get('correct', err.get('expected', ''))
                    golden_keys.add(f"{wrong}→{correct}")

        stats = {'added': 0, 'skipped_golden': 0}

        errors = report.get('errors', report if isinstance(report, list) else [])
        for err in errors:
            wrong = err.get('wrong', err.get('transcript', ''))
            correct = err.get('correct', err.get('original', ''))
            error_type = err.get('type', 'substitution')
            context = err.get('context', '')
            time_seconds = err.get('time_seconds')

            pattern_key = f"{wrong}→{correct}"

            if pattern_key in golden_keys:
                stats['skipped_golden'] += 1
                continue

            self.add_false_positive(wrong, correct, error_type, source, context, time_seconds)
            stats['added'] += 1

        return stats

    # =========================================================================
    # ЗАПРОСЫ
    # =========================================================================

    def get_top_patterns(
        self,
        limit: int = 20,
        status: str = 'active',
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Возвращает топ частых паттернов.

        Args:
            limit: Количество
            status: Фильтр по статусу (active, resolved, ignored, all)
            category: Фильтр по категории

        Returns:
            Список паттернов с их статистикой
        """
        query = 'SELECT * FROM patterns WHERE 1=1'
        params = []

        if status != 'all':
            query += ' AND status = ?'
            params.append(status)

        if category:
            query += ' AND category = ?'
            params.append(category)

        query += ' ORDER BY count DESC LIMIT ?'
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_pattern_by_key(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Получает паттерн по ключу"""
        cursor = self.conn.execute(
            'SELECT * FROM patterns WHERE pattern_key = ?',
            (pattern_key,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_occurrences(self, pattern_id: int) -> List[Dict[str, Any]]:
        """Получает все вхождения паттерна"""
        cursor = self.conn.execute(
            'SELECT * FROM occurrences WHERE pattern_id = ? ORDER BY created_at DESC',
            (pattern_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Поиск паттернов по wrong или correct"""
        cursor = self.conn.execute(
            '''SELECT * FROM patterns
               WHERE wrong LIKE ? OR correct LIKE ?
               ORDER BY count DESC''',
            (f'%{query}%', f'%{query}%')
        )
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # УПРАВЛЕНИЕ СТАТУСОМ
    # =========================================================================

    def mark_resolved(self, pattern_key: str, filter_name: str) -> bool:
        """
        Помечает паттерн как решённый.

        Args:
            pattern_key: Ключ паттерна ("wrong→correct")
            filter_name: Имя фильтра, который закрыл паттерн

        Returns:
            True если паттерн найден и обновлён
        """
        cursor = self.conn.execute(
            '''UPDATE patterns
               SET status = 'resolved', filter_applied = ?
               WHERE pattern_key = ?''',
            (filter_name, pattern_key)
        )
        self.conn.commit()

        if cursor.rowcount > 0:
            # Обновляем счётчик фильтра
            self._increment_filter_resolved(filter_name)
            return True
        return False

    def mark_ignored(self, pattern_key: str, reason: str = "") -> bool:
        """Помечает паттерн как игнорируемый"""
        cursor = self.conn.execute(
            '''UPDATE patterns
               SET status = 'ignored', filter_applied = ?
               WHERE pattern_key = ?''',
            (reason, pattern_key)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def reactivate(self, pattern_key: str) -> bool:
        """Возвращает паттерн в активное состояние"""
        cursor = self.conn.execute(
            '''UPDATE patterns
               SET status = 'active', filter_applied = NULL
               WHERE pattern_key = ?''',
            (pattern_key,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _increment_filter_resolved(self, filter_name: str):
        """Увеличивает счётчик решённых паттернов для фильтра"""
        now = datetime.now().isoformat()
        self.conn.execute(
            '''INSERT INTO filter_rules (name, patterns_resolved, created_at)
               VALUES (?, 1, ?)
               ON CONFLICT(name) DO UPDATE SET patterns_resolved = patterns_resolved + 1''',
            (filter_name, now)
        )
        self.conn.commit()

    # =========================================================================
    # СТАТИСТИКА
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает общую статистику"""
        cursor = self.conn.execute('''
            SELECT
                COUNT(*) as total_patterns,
                SUM(count) as total_occurrences,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
                SUM(CASE WHEN status = 'ignored' THEN 1 ELSE 0 END) as ignored
            FROM patterns
        ''')
        row = cursor.fetchone()

        # Статистика по категориям
        cursor = self.conn.execute('''
            SELECT category, COUNT(*) as count, SUM(count) as total_occurrences
            FROM patterns
            WHERE status = 'active'
            GROUP BY category
            ORDER BY count DESC
        ''')
        categories = {r['category']: {'patterns': r['count'], 'occurrences': r['total_occurrences']}
                      for r in cursor.fetchall()}

        # Статистика по источникам
        cursor = self.conn.execute('''
            SELECT source, COUNT(*) as count
            FROM occurrences
            GROUP BY source
            ORDER BY count DESC
        ''')
        sources = {r['source']: r['count'] for r in cursor.fetchall()}

        # Статистика по фильтрам
        cursor = self.conn.execute('''
            SELECT name, patterns_resolved
            FROM filter_rules
            ORDER BY patterns_resolved DESC
        ''')
        filters = {r['name']: r['patterns_resolved'] for r in cursor.fetchall()}

        return {
            'total_patterns': row['total_patterns'] or 0,
            'total_occurrences': row['total_occurrences'] or 0,
            'active': row['active'] or 0,
            'resolved': row['resolved'] or 0,
            'ignored': row['ignored'] or 0,
            'categories': categories,
            'sources': sources,
            'filters': filters,
            'db_path': str(self.db_path),
            'db_size_kb': self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0
        }

    def check_regressions(self, report_path: Path) -> List[Dict[str, Any]]:
        """
        Проверяет регрессии — resolved паттерны, которые снова появились.

        Args:
            report_path: Путь к новому filtered.json

        Returns:
            Список регрессий
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        regressions = []
        errors = report.get('errors', report if isinstance(report, list) else [])

        for err in errors:
            wrong = err.get('wrong', err.get('transcript', ''))
            correct = err.get('correct', err.get('original', ''))
            pattern_key = f"{wrong}→{correct}"

            pattern = self.get_pattern_by_key(pattern_key)
            if pattern and pattern['status'] == 'resolved':
                regressions.append({
                    'pattern_key': pattern_key,
                    'filter_applied': pattern['filter_applied'],
                    'error': err
                })

        return regressions

    # =========================================================================
    # ЭКСПОРТ/ИМПОРТ
    # =========================================================================

    def export_csv(self, output_path: Path):
        """Экспортирует паттерны в CSV"""
        cursor = self.conn.execute('SELECT * FROM patterns ORDER BY count DESC')

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pattern_key', 'wrong', 'correct', 'error_type',
                           'count', 'category', 'status', 'filter_applied',
                           'first_seen', 'last_seen'])

            for row in cursor.fetchall():
                writer.writerow([
                    row['pattern_key'], row['wrong'], row['correct'], row['error_type'],
                    row['count'], row['category'], row['status'], row['filter_applied'],
                    row['first_seen'], row['last_seen']
                ])

    def export_json(self, output_path: Path):
        """Экспортирует паттерны в JSON"""
        cursor = self.conn.execute('SELECT * FROM patterns ORDER BY count DESC')
        patterns = [dict(row) for row in cursor.fetchall()]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'patterns': patterns, 'stats': self.get_stats()}, f,
                     ensure_ascii=False, indent=2)

    # =========================================================================
    # ML-ПОДГОТОВКА (v2.0)
    # =========================================================================

    def mark_golden(self, golden_dir: Optional[Path] = None) -> Dict[str, int]:
        """
        Размечает паттерны из golden standard как is_golden=1.

        Args:
            golden_dir: Папка с файлами золотой_стандарт_*.json (по умолчанию TESTS_DIR)

        Returns:
            Статистика: {'marked': N, 'not_found': M, 'total_golden': K}
        """
        if golden_dir is None:
            golden_dir = TESTS_DIR

        golden_dir = Path(golden_dir)
        stats = {'marked': 0, 'not_found': 0, 'total_golden': 0}

        # Сброс всех меток
        self.conn.execute('UPDATE patterns SET is_golden = 0')

        # Загрузка golden standards
        golden_keys = set()
        for golden_file in golden_dir.glob('золотой_стандарт_*.json'):
            with open(golden_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for err in data.get('errors', []):
                    wrong = err.get('wrong', '')
                    correct = err.get('correct', '')
                    key = f"{wrong}→{correct}"
                    golden_keys.add(key)

        stats['total_golden'] = len(golden_keys)

        # Помечаем в БД
        for key in golden_keys:
            cursor = self.conn.execute(
                'UPDATE patterns SET is_golden = 1 WHERE pattern_key = ?',
                (key,)
            )
            if cursor.rowcount > 0:
                stats['marked'] += 1
            else:
                stats['not_found'] += 1

        self.conn.commit()
        return stats

    def update_morphology(self) -> Dict[str, int]:
        """
        Обновляет морфологические признаки для всех паттернов.

        Заполняет: lemma1, lemma2, pos1, pos2, aspect1, aspect2, levenshtein, same_lemma

        Returns:
            Статистика: {'updated': N, 'skipped': M}
        """
        if not HAS_MORPHOLOGY:
            return {'updated': 0, 'skipped': 0, 'error': 'morphology.py недоступен'}

        # Импортируем get_aspect если доступен
        try:
            from morphology import get_aspect
            has_aspect = True
        except ImportError:
            has_aspect = False
            def get_aspect(w):
                return None

        stats = {'updated': 0, 'skipped': 0}

        cursor = self.conn.execute('SELECT id, wrong, correct FROM patterns')
        rows = cursor.fetchall()

        for row in rows:
            pattern_id = row['id']
            wrong = row['wrong'] or ''
            correct = row['correct'] or ''

            try:
                lemma1 = get_lemma(wrong) if wrong else None
                lemma2 = get_lemma(correct) if correct else None
                pos1 = get_pos(wrong) if wrong else None
                pos2 = get_pos(correct) if correct else None
                aspect1 = get_aspect(wrong) if wrong and has_aspect else None
                aspect2 = get_aspect(correct) if correct and has_aspect else None

                # Левенштейн
                lev_dist = _levenshtein_distance(wrong.lower(), correct.lower()) if wrong and correct else None

                # Одна лемма?
                same_lemma = 1 if lemma1 and lemma2 and lemma1 == lemma2 else 0

                self.conn.execute('''
                    UPDATE patterns SET
                        lemma1 = ?, lemma2 = ?, pos1 = ?, pos2 = ?,
                        aspect1 = ?, aspect2 = ?, levenshtein = ?, same_lemma = ?
                    WHERE id = ?
                ''', (lemma1, lemma2, pos1, pos2, aspect1, aspect2, lev_dist, same_lemma, pattern_id))

                stats['updated'] += 1

            except Exception as e:
                stats['skipped'] += 1

        self.conn.commit()
        return stats

    def export_features_csv(self, output_path: Path) -> int:
        """
        Экспортирует признаки для ML в CSV.

        Returns:
            Количество экспортированных записей
        """
        cursor = self.conn.execute('''
            SELECT pattern_key, wrong, correct, error_type, count, category,
                   is_golden, lemma1, lemma2, pos1, pos2, aspect1, aspect2,
                   levenshtein, same_lemma, ml_score
            FROM patterns
            ORDER BY count DESC
        ''')

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pattern_key', 'wrong', 'correct', 'error_type', 'count', 'category',
                'is_golden', 'lemma1', 'lemma2', 'pos1', 'pos2', 'aspect1', 'aspect2',
                'levenshtein', 'same_lemma', 'ml_score',
                # Вычисляемые признаки
                'len_wrong', 'len_correct', 'len_diff'
            ])

            count = 0
            for row in cursor.fetchall():
                wrong = row['wrong'] or ''
                correct = row['correct'] or ''
                writer.writerow([
                    row['pattern_key'], wrong, correct, row['error_type'],
                    row['count'], row['category'], row['is_golden'],
                    row['lemma1'], row['lemma2'], row['pos1'], row['pos2'],
                    row['aspect1'], row['aspect2'], row['levenshtein'],
                    row['same_lemma'], row['ml_score'],
                    len(wrong), len(correct), abs(len(wrong) - len(correct))
                ])
                count += 1

        return count

    def get_ml_stats(self) -> Dict[str, Any]:
        """Возвращает статистику для ML."""
        cursor = self.conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_golden = 1 THEN 1 ELSE 0 END) as golden,
                SUM(CASE WHEN is_golden = 0 THEN 1 ELSE 0 END) as non_golden,
                SUM(CASE WHEN lemma1 IS NOT NULL THEN 1 ELSE 0 END) as with_morph,
                SUM(CASE WHEN same_lemma = 1 THEN 1 ELSE 0 END) as same_lemma_count
            FROM patterns
        ''')
        row = cursor.fetchone()

        return {
            'total_patterns': row['total'] or 0,
            'golden_count': row['golden'] or 0,
            'non_golden_count': row['non_golden'] or 0,
            'with_morphology': row['with_morph'] or 0,
            'same_lemma_count': row['same_lemma_count'] or 0,
            'class_balance': f"{row['golden'] or 0}:{row['non_golden'] or 0}"
        }

    def migrate_from_json(self, json_path: Path) -> Dict[str, int]:
        """
        Миграция из существующего JSON-файла false_positives_tracker.json

        Args:
            json_path: Путь к JSON-файлу

        Returns:
            Статистика миграции
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stats = {'migrated': 0, 'skipped': 0}

        false_positives = data.get('false_positives', {})

        for pattern_key, fp_data in false_positives.items():
            wrong = fp_data.get('wrong', '')
            correct = fp_data.get('correct', '')
            error_type = fp_data.get('type', 'substitution')
            count = fp_data.get('count', 1)
            context = fp_data.get('context', '')
            sources = fp_data.get('sources', [])
            first_seen = fp_data.get('first_seen', datetime.now().isoformat())
            last_seen = fp_data.get('last_seen', datetime.now().isoformat())

            # Проверяем, существует ли уже
            existing = self.get_pattern_by_key(pattern_key)
            if existing:
                stats['skipped'] += 1
                continue

            # Классифицируем
            category = classify_pattern(wrong, correct, error_type)

            # Добавляем паттерн
            cursor = self.conn.execute(
                '''INSERT INTO patterns
                   (wrong, correct, error_type, pattern_key, count, category, status, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)''',
                (wrong, correct, error_type, pattern_key, count, category, first_seen, last_seen)
            )
            pattern_id = cursor.lastrowid

            # Добавляем вхождения для каждого источника
            for source in sources:
                self.conn.execute(
                    '''INSERT INTO occurrences (pattern_id, source, context, created_at)
                       VALUES (?, ?, ?, ?)''',
                    (pattern_id, source, context, first_seen)
                )

            stats['migrated'] += 1

        self.conn.commit()
        return stats

    def close(self):
        """Закрывает соединение с БД"""
        if self.conn:
            self.conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='False Positives Database CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python false_positives_db.py stats                    # Статистика
  python false_positives_db.py top 20                   # Топ-20 паттернов
  python false_positives_db.py top 20 --category grammar_ending
  python false_positives_db.py add report.json --source 01_yandex
  python false_positives_db.py resolve "живем→живы" --filter grammar_ending
  python false_positives_db.py migrate                  # Миграция из JSON
  python false_positives_db.py export --csv output.csv
  python false_positives_db.py check-regressions report.json
        """
    )

    parser.add_argument('--version', '-V', action='store_true', help='Версия')

    subparsers = parser.add_subparsers(dest='command', help='Команда')

    # stats
    subparsers.add_parser('stats', help='Статистика базы')

    # top
    top_parser = subparsers.add_parser('top', help='Топ паттернов')
    top_parser.add_argument('limit', type=int, nargs='?', default=20, help='Количество')
    top_parser.add_argument('--category', '-c', help='Фильтр по категории')
    top_parser.add_argument('--status', '-s', default='active', help='Статус (active/resolved/ignored/all)')

    # add
    add_parser = subparsers.add_parser('add', help='Добавить из отчёта')
    add_parser.add_argument('report', help='Путь к filtered.json')
    add_parser.add_argument('--source', '-s', required=True, help='Идентификатор источника')
    add_parser.add_argument('--golden', '-g', help='Путь к golden standard')

    # resolve
    resolve_parser = subparsers.add_parser('resolve', help='Пометить как решённый')
    resolve_parser.add_argument('pattern', help='Ключ паттерна (wrong→correct)')
    resolve_parser.add_argument('--filter', '-f', required=True, help='Имя фильтра')

    # migrate
    migrate_parser = subparsers.add_parser('migrate', help='Миграция из JSON')
    migrate_parser.add_argument('--json', '-j', help='Путь к JSON (по умолчанию Словари/false_positives_tracker.json)')

    # export
    export_parser = subparsers.add_parser('export', help='Экспорт')
    export_parser.add_argument('--csv', help='Экспорт в CSV')
    export_parser.add_argument('--json', help='Экспорт в JSON')

    # check-regressions
    regr_parser = subparsers.add_parser('check-regressions', help='Проверить регрессии')
    regr_parser.add_argument('report', help='Путь к новому filtered.json')

    # search
    search_parser = subparsers.add_parser('search', help='Поиск паттернов')
    search_parser.add_argument('query', help='Поисковый запрос')

    # update-morph (v2.0)
    subparsers.add_parser('update-morph', help='Обновить морфологические признаки')

    # mark-golden (v2.0)
    golden_parser = subparsers.add_parser('mark-golden', help='Разметить golden standard')
    golden_parser.add_argument('--dir', '-d', help='Папка с golden standard файлами')

    # ml-stats (v2.0)
    subparsers.add_parser('ml-stats', help='Статистика для ML')

    # export-features (v2.0)
    features_parser = subparsers.add_parser('export-features', help='Экспорт признаков для ML')
    features_parser.add_argument('output', help='Путь к выходному CSV файлу')

    args = parser.parse_args()

    if args.version:
        print(f"False Positives DB v{VERSION} ({VERSION_DATE})")
        return

    db = FalsePositivesDB()

    try:
        if args.command == 'stats':
            stats = db.get_stats()
            print("\n=== Статистика False Positives DB ===")
            print(f"Всего паттернов: {stats['total_patterns']}")
            print(f"Всего вхождений: {stats['total_occurrences']}")
            print(f"  Активных: {stats['active']}")
            print(f"  Решённых: {stats['resolved']}")
            print(f"  Игнорируемых: {stats['ignored']}")
            print(f"\nПо категориям:")
            for cat, data in stats['categories'].items():
                print(f"  {cat}: {data['patterns']} паттернов, {data['occurrences']} вхождений")
            print(f"\nПо источникам:")
            for src, count in stats['sources'].items():
                print(f"  {src}: {count}")
            if stats['filters']:
                print(f"\nПо фильтрам (решённых):")
                for flt, count in stats['filters'].items():
                    print(f"  {flt}: {count}")
            print(f"\nРазмер БД: {stats['db_size_kb']:.1f} KB")
            print(f"Путь: {stats['db_path']}")

        elif args.command == 'top':
            patterns = db.get_top_patterns(args.limit, args.status, args.category)
            print(f"\n=== Топ-{args.limit} паттернов ({args.status}) ===\n")
            for i, p in enumerate(patterns, 1):
                print(f"{i:2}. [{p['count']:3}] {p['pattern_key']}")
                print(f"      Категория: {p['category']}, Тип: {p['error_type']}")
                if p['filter_applied']:
                    print(f"      Фильтр: {p['filter_applied']}")

        elif args.command == 'add':
            golden_path = Path(args.golden) if args.golden else None
            stats = db.add_from_report(Path(args.report), args.source, golden_path)
            print(f"✓ Добавлено: {stats['added']}")
            print(f"  Пропущено (golden): {stats['skipped_golden']}")

        elif args.command == 'resolve':
            if db.mark_resolved(args.pattern, args.filter):
                print(f"✓ Паттерн '{args.pattern}' помечен как resolved (фильтр: {args.filter})")
            else:
                print(f"✗ Паттерн '{args.pattern}' не найден")

        elif args.command == 'migrate':
            json_path = Path(args.json) if args.json else DICTS_DIR / 'false_positives_tracker.json'
            if not json_path.exists():
                print(f"✗ Файл не найден: {json_path}")
                return
            stats = db.migrate_from_json(json_path)
            print(f"✓ Миграция завершена:")
            print(f"  Мигрировано: {stats['migrated']}")
            print(f"  Пропущено (дубли): {stats['skipped']}")

        elif args.command == 'export':
            if args.csv:
                db.export_csv(Path(args.csv))
                print(f"✓ Экспортировано в CSV: {args.csv}")
            if args.json:
                db.export_json(Path(args.json))
                print(f"✓ Экспортировано в JSON: {args.json}")

        elif args.command == 'check-regressions':
            regressions = db.check_regressions(Path(args.report))
            if regressions:
                print(f"\n⚠ Найдено {len(regressions)} регрессий:\n")
                for r in regressions:
                    print(f"  {r['pattern_key']} (был фильтр: {r['filter_applied']})")
            else:
                print("✓ Регрессий не найдено")

        elif args.command == 'search':
            results = db.search_patterns(args.query)
            print(f"\n=== Поиск: '{args.query}' ({len(results)} результатов) ===\n")
            for p in results:
                print(f"  [{p['count']:3}] {p['pattern_key']} ({p['status']}, {p['category']})")

        elif args.command == 'update-morph':
            print("Обновление морфологических признаков...")
            stats = db.update_morphology()
            print(f"✓ Обновлено: {stats['updated']}")
            print(f"  Пропущено: {stats['skipped']}")
            if 'error' in stats:
                print(f"  Ошибка: {stats['error']}")

        elif args.command == 'mark-golden':
            golden_dir = Path(args.dir) if args.dir else None
            print("Разметка golden standard...")
            stats = db.mark_golden(golden_dir)
            print(f"✓ Всего golden ошибок: {stats['total_golden']}")
            print(f"  Найдено в БД: {stats['marked']}")
            print(f"  Не найдено: {stats['not_found']}")

        elif args.command == 'ml-stats':
            stats = db.get_ml_stats()
            print("\n=== ML-статистика ===")
            print(f"Всего паттернов: {stats['total_patterns']}")
            print(f"Golden (реальные ошибки): {stats['golden_count']}")
            print(f"Non-golden (ложные): {stats['non_golden_count']}")
            print(f"С морфологией: {stats['with_morphology']}")
            print(f"Одинаковая лемма: {stats['same_lemma_count']}")
            print(f"Баланс классов: {stats['class_balance']}")

        elif args.command == 'export-features':
            count = db.export_features_csv(Path(args.output))
            print(f"✓ Экспортировано {count} записей в {args.output}")

        else:
            parser.print_help()

    finally:
        db.close()


if __name__ == '__main__':
    main()
