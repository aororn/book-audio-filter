#!/usr/bin/env python3
"""
Database Writer for Filter Engine v1.0

Интеграция системы фильтрации с БД:
- Запись результатов фильтрации напрямую в БД
- Автоматическое отслеживание истории изменений
- Генерация JSON из БД (экспорт)

v1.0 (2026-01-31): Начальная версия — интеграция engine.py с БД
"""

VERSION = '2.1.0'  # v2.1: Интеграция error_normalizer для унификации полей

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Путь к БД — ЕДИНЫЙ источник из config.py
import sys
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from config import FALSE_POSITIVES_DB, TESTS_DIR
    from version import PROJECT_VERSION, FILTER_ENGINE_VERSION
    DB_PATH = FALSE_POSITIVES_DB
except ImportError:
    DB_PATH = Path(__file__).parent.parent.parent / 'Словари' / 'false_positives.db'
    TESTS_DIR = Path(__file__).parent.parent.parent / 'Тесты'
    PROJECT_VERSION = 'unknown'
    FILTER_ENGINE_VERSION = 'unknown'

# v2.1: Импорт error_normalizer для унификации полей
try:
    from error_normalizer import (
        get_original_word, get_transcript_word, get_time_seconds,
        get_error_type, get_context, normalize_word, errors_match,
        is_error_in_list, make_error_key,
    )
    HAS_NORMALIZER = True
except ImportError:
    HAS_NORMALIZER = False
    # Fallback функции
    def get_original_word(e): return e.get('original', e.get('correct', e.get('from_book', '')))
    def get_transcript_word(e): return e.get('transcript', e.get('wrong', e.get('word', '')))
    def get_time_seconds(e): return e.get('time_seconds', e.get('time', 0))
    def get_error_type(e): return e.get('type', e.get('error_type', 'substitution'))
    def get_context(e): return e.get('context', '')
    def normalize_word(w): return w.lower().replace('ё', 'е').strip() if w else ''

# Импорты для вычисления полей
try:
    from morphology import get_lemma, get_pos
    from filters.semantic_manager import get_similarity
    from filters.frequency_manager import FrequencyManager
    from filters.comparison import levenshtein_distance
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False
    def get_lemma(w): return w.lower() if w else ''
    def get_pos(w): return ''
    def get_similarity(w1, w2): return 0.0
    def levenshtein_distance(w1, w2): return abs(len(w1 or '') - len(w2 or ''))

# Глобальный менеджер частотности (ленивая инициализация)
_frequency_manager = None

def _get_frequency_manager():
    global _frequency_manager
    if _frequency_manager is None and HAS_MORPHOLOGY:
        _frequency_manager = FrequencyManager()
    return _frequency_manager


@dataclass
class FilterResult:
    """Результат фильтрации одной ошибки — ВСЕ поля БД"""
    error_id: str
    wrong: str
    correct: str
    error_type: str
    chapter: int
    time_seconds: float
    is_filtered: bool
    filter_reason: Optional[str]
    is_golden: bool = False
    context: str = ''

    # Контексты
    transcript_context: str = ''
    pos_transcript: Optional[int] = None
    pos_original: Optional[int] = None

    # Фонетика
    phonetic_similarity: float = 0.0
    levenshtein: int = 0

    # Морфология (v2.0)
    lemma_wrong: str = ''
    lemma_correct: str = ''
    pos_wrong: str = ''
    pos_correct: str = ''
    same_lemma: int = 0
    same_pos: int = 0

    # Семантика (v2.0)
    semantic_similarity: float = 0.0

    # Частотность (v2.0)
    frequency_wrong: int = 0
    frequency_correct: int = 0

    # Временные окна
    time_end_seconds: float = 0.0
    window_start: float = 0.0
    window_end: float = 0.0
    context_start: float = 0.0
    context_end: float = 0.0

    # Контексты как JSON
    context_transcript: str = ''
    context_original: str = ''
    context_normalized: str = ''
    context_aligned: str = ''

    # Связи
    linked_errors: str = ''
    link_type: str = ''

    @classmethod
    def from_error_dict(cls, error: Dict, chapter: int, is_filtered: bool,
                        filter_reason: Optional[str], is_golden: bool) -> 'FilterResult':
        """
        Создаёт FilterResult из словаря ошибки, вычисляя ВСЕ поля.

        v2.1: Использует error_normalizer для унификации полей.
        Принимает ошибки с ЛЮБЫМИ названиями полей:
        - JSON: original, transcript
        - БД: correct, wrong
        - Golden: correct/original, wrong/transcript
        """
        # v2.1: Унифицированное извлечение полей через error_normalizer
        wrong = get_transcript_word(error)  # transcript/wrong/word
        correct = get_original_word(error)  # original/correct/from_book
        error_type = get_error_type(error)  # type/error_type
        time_sec = get_time_seconds(error)  # time/time_seconds

        # Морфология
        lemma_w = get_lemma(wrong) if wrong else ''
        lemma_c = get_lemma(correct) if correct else ''
        pos_w = get_pos(wrong) if wrong else ''
        pos_c = get_pos(correct) if correct else ''
        same_lemma = 1 if (lemma_w and lemma_c and lemma_w == lemma_c) else 0
        same_pos = 1 if (pos_w and pos_c and pos_w == pos_c) else 0

        # Семантика (только для substitution)
        sem = 0.0
        if wrong and correct and error_type == 'substitution':
            sem = get_similarity(wrong, correct)

        # Частотность
        freq_mgr = _get_frequency_manager()
        freq_w = freq_mgr.get_frequency(wrong) if freq_mgr and wrong else 0
        freq_c = freq_mgr.get_frequency(correct) if freq_mgr and correct else 0

        # Фонетика
        phon_sim = error.get('phonetic_similarity', error.get('similarity', 0))
        lev = levenshtein_distance(wrong, correct) if wrong and correct else 0

        return cls(
            error_id=error.get('error_id', str(uuid.uuid4())[:8]),
            wrong=wrong,
            correct=correct,
            error_type=error_type,
            chapter=chapter,
            time_seconds=time_sec,
            is_filtered=is_filtered,
            filter_reason=filter_reason,
            is_golden=is_golden,
            context=error.get('context', ''),
            transcript_context=error.get('transcript_context', ''),
            pos_transcript=error.get('marker_pos'),
            pos_original=error.get('pos_original'),
            phonetic_similarity=phon_sim,
            levenshtein=lev,
            lemma_wrong=lemma_w,
            lemma_correct=lemma_c,
            pos_wrong=pos_w,
            pos_correct=pos_c,
            same_lemma=same_lemma,
            same_pos=same_pos,
            semantic_similarity=sem,
            frequency_wrong=freq_w,
            frequency_correct=freq_c,
            time_end_seconds=error.get('time_end', 0),
            window_start=error.get('window_start', 0),
            window_end=error.get('window_end', 0),
            context_start=error.get('context_start', 0),
            context_end=error.get('context_end', 0),
            context_transcript=error.get('context_transcript', ''),
            context_original=error.get('context_original', ''),
            context_normalized=error.get('context_normalized', ''),
            context_aligned=error.get('context_aligned', ''),
            linked_errors=error.get('linked_errors', ''),
            link_type=error.get('link_type', ''),
        )


class DatabaseWriter:
    """
    Записывает результаты фильтрации в БД.

    Использование:
        writer = DatabaseWriter()
        writer.start_session(chapter=1)

        for error in errors:
            result = FilterResult(...)
            writer.write_error(result)

        writer.finish_session()
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.run_id: str = ''
        self.run_timestamp: str = ''
        self.chapter: int = 0
        self.changes: List[Dict] = []
        self._existing_errors: Dict[str, Dict] = {}

    def connect(self):
        """Подключение к БД"""
        if self.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

    def close(self):
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def start_session(self, chapter: int):
        """Начинает сессию записи для главы"""
        self.connect()
        self.run_id = str(uuid.uuid4())[:12]
        self.run_timestamp = datetime.now().isoformat()
        self.chapter = chapter
        self.changes = []

        # Загружаем существующие ошибки для сравнения
        self._load_existing_errors(chapter)

        # Считаем ошибки до
        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE chapter = ?', (chapter,))
        errors_before = cur.fetchone()[0]

        # Записываем начало сессии
        self.conn.execute('''
            INSERT INTO sync_runs (
                run_id, run_timestamp, project_version, filter_version,
                chapters_processed, is_reset, total_errors_before
            ) VALUES (?, ?, ?, ?, ?, 0, ?)
        ''', (
            self.run_id,
            self.run_timestamp,
            PROJECT_VERSION,
            FILTER_ENGINE_VERSION,
            json.dumps([chapter]),
            errors_before,
        ))
        self.conn.commit()

        return self.run_id

    def _load_existing_errors(self, chapter: int):
        """Загружает существующие ошибки для diff"""
        self._existing_errors = {}
        cur = self.conn.execute('''
            SELECT error_id, wrong, correct, error_type, time_seconds,
                   is_filtered, filter_reason, is_golden
            FROM errors
            WHERE chapter = ?
        ''', (chapter,))

        for row in cur:
            key = self._make_key(row[1], row[2], row[3], row[4])
            self._existing_errors[key] = {
                'error_id': row[0],
                'wrong': row[1],
                'correct': row[2],
                'error_type': row[3],
                'time_seconds': row[4],
                'is_filtered': row[5],
                'filter_reason': row[6],
                'is_golden': row[7],
            }

    @staticmethod
    def _make_key(wrong: str, correct: str, error_type: str, time_seconds: float) -> str:
        """
        Создаёт ключ для идентификации ошибки.

        v2.1: Нормализует слова для корректного сравнения.
        """
        # v2.1: Нормализуем слова для консистентности
        wrong_norm = _normalize_word_local(wrong)
        correct_norm = _normalize_word_local(correct)
        time_key = round(time_seconds, 1) if time_seconds else 0
        return f"{wrong_norm}|{correct_norm}|{error_type}|{time_key}"

    def write_error(self, result: FilterResult):
        """Записывает одну ошибку в БД с отслеживанием изменений"""
        key = self._make_key(result.wrong, result.correct, result.error_type, result.time_seconds)

        # Определяем действие
        old_data = self._existing_errors.get(key)
        action = None

        if old_data:
            # Ошибка существует — проверяем изменения
            old_filtered = old_data.get('is_filtered', 0)
            new_filtered = 1 if result.is_filtered else 0

            if old_filtered != new_filtered:
                action = 'filtered' if new_filtered else 'unfiltered'
            elif old_filtered and new_filtered:
                old_reason = old_data.get('filter_reason') or ''
                new_reason = result.filter_reason or ''
                if old_reason != new_reason:
                    action = 'filter_reason_changed'

            # Проверяем golden
            old_golden = old_data.get('is_golden', 0)
            new_golden = 1 if result.is_golden else 0
            if old_golden != new_golden:
                action = 'golden_added' if new_golden else 'golden_removed'
        else:
            # Новая ошибка
            action = 'created'

        # Записываем изменение в историю
        if action:
            self._record_history(action, result, old_data)
            self.changes.append({'action': action, 'error_id': result.error_id})

        # Обновляем/вставляем ошибку
        self._upsert_error(result)

        # Отмечаем как обработанную
        if key in self._existing_errors:
            self._existing_errors[key]['_processed'] = True

    def _record_history(self, action: str, result: FilterResult, old_data: Optional[Dict]):
        """Записывает изменение в историю"""
        self.conn.execute('''
            INSERT INTO error_history (
                run_id, run_timestamp, project_version, filter_version,
                error_id, chapter, time_seconds, wrong, correct, error_type,
                action,
                old_is_filtered, new_is_filtered,
                old_filter_reason, new_filter_reason,
                old_is_golden, new_is_golden,
                context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.run_id,
            self.run_timestamp,
            PROJECT_VERSION,
            FILTER_ENGINE_VERSION,
            result.error_id,
            result.chapter,
            result.time_seconds,
            result.wrong,
            result.correct,
            result.error_type,
            action,
            (old_data or {}).get('is_filtered'),
            1 if result.is_filtered else 0,
            (old_data or {}).get('filter_reason'),
            result.filter_reason,
            (old_data or {}).get('is_golden'),
            1 if result.is_golden else 0,
            result.context,
            datetime.now().isoformat(),
        ))

    def _upsert_error(self, result: FilterResult):
        """Вставляет или обновляет ошибку — ВСЕ поля v2.0"""
        now = datetime.now().isoformat()

        self.conn.execute('''
            INSERT OR REPLACE INTO errors (
                error_id, wrong, correct, error_type, chapter,
                time_seconds, time_end_seconds, time_label,
                window_start, window_end, context_start, context_end,
                pos_transcript, pos_original,
                context, transcript_context,
                context_transcript, context_normalized, context_original, context_aligned,
                linked_errors, link_type,
                is_golden, is_filtered, filter_reason,
                lemma_wrong, lemma_correct, pos_wrong, pos_correct, same_lemma, same_pos,
                semantic_similarity, frequency_wrong, frequency_correct,
                phonetic_similarity, levenshtein,
                created_at, updated_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)
        ''', (
            result.error_id,
            result.wrong,
            result.correct,
            result.error_type,
            result.chapter,
            result.time_seconds,
            result.time_end_seconds,
            self._seconds_to_label(result.time_seconds),
            result.window_start,
            result.window_end,
            result.context_start,
            result.context_end,
            result.pos_transcript,
            result.pos_original,
            result.context,
            result.transcript_context,
            result.context_transcript,
            result.context_normalized,
            result.context_original,
            result.context_aligned,
            result.linked_errors,
            result.link_type,
            1 if result.is_golden else 0,
            1 if result.is_filtered else 0,
            result.filter_reason,
            result.lemma_wrong,
            result.lemma_correct,
            result.pos_wrong,
            result.pos_correct,
            result.same_lemma,
            result.same_pos,
            result.semantic_similarity,
            result.frequency_wrong,
            result.frequency_correct,
            result.phonetic_similarity,
            result.levenshtein,
            now,
            now,
        ))

    @staticmethod
    def _seconds_to_label(seconds: float) -> str:
        """Конвертирует секунды в MM:SS"""
        if not seconds:
            return ""
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"

    def finish_session(self) -> Dict[str, int]:
        """Завершает сессию, обрабатывает удалённые ошибки"""
        # Обрабатываем ошибки, которые исчезли
        for key, old_data in self._existing_errors.items():
            if not old_data.get('_processed'):
                # Ошибка исчезла (больше не находится выравниванием)
                self._record_history_deleted(old_data)
                self.changes.append({'action': 'deleted', 'error_id': old_data.get('error_id')})

        # Считаем статистику
        action_counts = {}
        for change in self.changes:
            action = change['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        # Считаем ошибки после
        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE chapter = ?', (self.chapter,))
        errors_after = cur.fetchone()[0]

        # Обновляем sync_runs
        self.conn.execute('''
            UPDATE sync_runs SET
                run_finished = ?,
                total_errors_after = ?,
                errors_created = ?,
                errors_deleted = ?,
                errors_filtered = ?,
                errors_unfiltered = ?,
                golden_added = ?,
                golden_removed = ?
            WHERE run_id = ?
        ''', (
            datetime.now().isoformat(),
            errors_after,
            action_counts.get('created', 0),
            action_counts.get('deleted', 0),
            action_counts.get('filtered', 0),
            action_counts.get('unfiltered', 0),
            action_counts.get('golden_added', 0),
            action_counts.get('golden_removed', 0),
            self.run_id,
        ))

        self.conn.commit()

        return action_counts

    def _record_history_deleted(self, old_data: Dict):
        """Записывает удаление ошибки"""
        self.conn.execute('''
            INSERT INTO error_history (
                run_id, run_timestamp, project_version, filter_version,
                error_id, chapter, time_seconds, wrong, correct, error_type,
                action,
                old_is_filtered, new_is_filtered,
                old_filter_reason, new_filter_reason,
                old_is_golden, new_is_golden,
                context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'deleted', ?, NULL, ?, NULL, ?, NULL, '', ?)
        ''', (
            self.run_id,
            self.run_timestamp,
            PROJECT_VERSION,
            FILTER_ENGINE_VERSION,
            old_data.get('error_id', ''),
            self.chapter,
            old_data.get('time_seconds', 0),
            old_data.get('wrong', ''),
            old_data.get('correct', ''),
            old_data.get('error_type', ''),
            old_data.get('is_filtered'),
            old_data.get('filter_reason'),
            old_data.get('is_golden'),
            datetime.now().isoformat(),
        ))


class DatabaseExporter:
    """
    Экспортирует данные из БД в JSON формат.

    Позволяет генерировать filtered.json из БД
    (БД становится источником правды).
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Подключение к БД"""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row

    def export_chapter(self, chapter: int) -> Dict:
        """
        Экспортирует главу в формат filtered.json

        Returns:
            Dict совместимый с форматом filtered.json
        """
        self.connect()

        # Получаем все ошибки главы
        cur = self.conn.execute('''
            SELECT * FROM errors WHERE chapter = ?
            ORDER BY time_seconds
        ''', (chapter,))

        all_errors = [dict(row) for row in cur.fetchall()]

        # Разделяем на filtered и remaining
        filtered_errors = [e for e in all_errors if e.get('is_filtered')]
        remaining_errors = [e for e in all_errors if not e.get('is_filtered')]

        # Считаем статистику фильтрации
        filter_stats = {}
        for e in filtered_errors:
            reason = e.get('filter_reason') or 'unknown'
            filter_stats[reason] = filter_stats.get(reason, 0) + 1

        # Формируем отчёт
        report = {
            'errors': [self._error_to_json(e) for e in remaining_errors],
            'total_errors': len(remaining_errors),
            'filtered_count': len(filtered_errors),
            'filter_stats': filter_stats,
            'filtered_errors_detail': [self._error_to_json(e) for e in filtered_errors],
            'metadata': {
                'source': 'database',
                'db_version': '2.1',
                'exported_at': datetime.now().isoformat(),
                'chapter': chapter,
            },
            'filter_metadata': {
                'version': FILTER_ENGINE_VERSION,
                'original_errors': len(all_errors),
                'real_errors': len(remaining_errors),
                'filtered_errors': len(filtered_errors),
                'filter_breakdown': {
                    reason: {'count': count}
                    for reason, count in filter_stats.items()
                },
            },
        }

        return report

    def _error_to_json(self, db_row: Dict) -> Dict:
        """Конвертирует запись БД в формат JSON ошибки"""
        return {
            'transcript': db_row.get('wrong', ''),
            'original': db_row.get('correct', ''),
            'type': db_row.get('error_type', 'substitution'),
            'time': db_row.get('time_seconds', 0),
            'time_label': db_row.get('time_label', ''),
            'context': db_row.get('context', ''),
            'phonetic_similarity': db_row.get('phonetic_similarity', 0),
            'similarity': db_row.get('phonetic_similarity', 0),
            'filter_reason': db_row.get('filter_reason'),
            'is_golden': bool(db_row.get('is_golden')),
            # Добавляем ID для отслеживания
            'error_id': db_row.get('error_id', ''),
        }

    def export_to_file(self, chapter: int, output_path: Path):
        """Экспортирует главу в JSON файл"""
        report = self.export_chapter(chapter)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return output_path


# =============================================================================
# Утилиты для интеграции с engine.py
# =============================================================================

def _load_golden_for_chapter(chapter: int) -> List[Dict]:
    """Загружает golden ошибки для главы"""
    try:
        golden_path = TESTS_DIR / f'золотой_стандарт_глава{chapter}.json'
        if golden_path.exists():
            with open(golden_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('errors', [])
    except Exception:
        pass
    return []


def write_filter_results(
    chapter: int,
    all_errors: List[Dict],
    filtered: List[Dict],
    removed: List[Dict],
    golden_set: set = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Записывает результаты фильтрации в БД с ПОЛНЫМ вычислением всех полей.

    Вызывается из engine.py::filter_report() после фильтрации.

    Args:
        chapter: Номер главы
        all_errors: Все ошибки до фильтрации
        filtered: Оставшиеся ошибки (реальные)
        removed: Отфильтрованные ошибки
        golden_set: Множество golden ошибок (опционально)

    Returns:
        (run_id, action_counts)
    """
    writer = DatabaseWriter()
    run_id = writer.start_session(chapter)

    # Загружаем golden если не передан
    golden_list = []
    if golden_set is None:
        golden_list = _load_golden_for_chapter(chapter)
    elif isinstance(golden_set, (list, set)):
        golden_list = list(golden_set)

    # Записываем оставшиеся (реальные) ошибки — с полным вычислением полей
    for error in filtered:
        is_golden = _is_golden(error, golden_list)
        result = FilterResult.from_error_dict(
            error=error,
            chapter=chapter,
            is_filtered=False,
            filter_reason=None,
            is_golden=is_golden,
        )
        writer.write_error(result)

    # Записываем отфильтрованные ошибки — с полным вычислением полей
    for error in removed:
        is_golden = _is_golden(error, golden_list)
        result = FilterResult.from_error_dict(
            error=error,
            chapter=chapter,
            is_filtered=True,
            filter_reason=error.get('filter_reason', 'unknown'),
            is_golden=is_golden,
        )
        writer.write_error(result)

    action_counts = writer.finish_session()
    writer.close()

    return run_id, action_counts


def _normalize_word_local(word: str) -> str:
    """Нормализует слово для сравнения (локальная версия)"""
    return normalize_word(word) if HAS_NORMALIZER else (word.lower().replace('ё', 'е').strip() if word else '')


def _is_golden(error: Dict, golden_list: List[Dict]) -> bool:
    """
    Проверяет, является ли ошибка golden.

    v2.1: Использует error_normalizer для унификации полей.
    Работает с ЛЮБЫМИ названиями полей (original/correct, transcript/wrong).
    """
    if not golden_list:
        return False

    # v2.1: Унифицированное извлечение через error_normalizer
    wrong = _normalize_word_local(get_transcript_word(error))
    correct = _normalize_word_local(get_original_word(error))
    time_sec = get_time_seconds(error)

    TIME_TOLERANCE = 15

    for g in golden_list:
        if not isinstance(g, dict):
            continue

        # v2.1: Golden файлы тоже могут иметь разные форматы
        g_wrong = _normalize_word_local(get_transcript_word(g))
        g_correct = _normalize_word_local(get_original_word(g))
        g_time = get_time_seconds(g)

        # Проверяем по времени
        if abs(g_time - time_sec) > TIME_TOLERANCE:
            continue

        # Проверяем совпадение слов
        if correct and g_correct and correct == g_correct:
            return True
        if wrong and g_wrong and wrong == g_wrong:
            return True

        # Специальные случаи (insertion/deletion)
        if g_correct == '' and correct == '' and g_wrong and wrong:
            if g_wrong == wrong:
                return True
        if g_wrong == '' and wrong == '' and g_correct and correct:
            if g_correct == correct:
                return True

    return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database Writer v1.0')
    parser.add_argument('--export', type=int, help='Экспортировать главу в JSON')
    parser.add_argument('--output', type=str, help='Путь для экспорта')
    parser.add_argument('--info', action='store_true', help='Информация о БД')

    args = parser.parse_args()

    print(f"Database Writer v{VERSION}")
    print("=" * 60)

    if args.export:
        exporter = DatabaseExporter()

        if args.output:
            output = Path(args.output)
        else:
            output = Path(f'chapter_{args.export:02d}_exported.json')

        exporter.export_to_file(args.export, output)
        print(f"[OK] Экспортировано: {output}")

    elif args.info:
        writer = DatabaseWriter()
        writer.connect()

        # Статистика по главам
        cur = writer.conn.execute('''
            SELECT chapter, COUNT(*) as total,
                   SUM(is_filtered) as filtered,
                   SUM(is_golden) as golden
            FROM errors
            GROUP BY chapter
            ORDER BY chapter
        ''')

        print("\nСтатистика по главам:")
        for row in cur:
            print(f"  Глава {row[0]}: {row[1]} ошибок, {row[2]} filtered, {row[3]} golden")

        # История прогонов
        cur = writer.conn.execute('''
            SELECT run_id, run_timestamp, total_errors_after,
                   errors_created, errors_filtered
            FROM sync_runs
            ORDER BY run_timestamp DESC
            LIMIT 5
        ''')

        print("\nПоследние прогоны:")
        for row in cur:
            print(f"  {row[0][:8]}  {row[1][:19]}  errors={row[2]}  +{row[3]}  filt={row[4]}")

        writer.close()

    else:
        parser.print_help()
