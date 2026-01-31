#!/usr/bin/env python3
"""
Populate False Positives Database v3.1

Расширенная версия с:
- Контекстами из 4 источников (original, transcript, normalized, aligned)
- Позициями в файлах (pos_transcript, pos_original, pos_normalized)
- Временными окнами (window_start, window_end, context_start, context_end)
- Детекцией merge/split паттернов (linked_errors, link_type)
- Связями между ошибками (таблица error_links)
- **v3.1: История изменений (diff-логика, таблица error_history)**

Использование:
    python populate_db_v3.py                  # Заполнить БД (с историей)
    python populate_db_v3.py --stats          # Показать статистику
    python populate_db_v3.py --reset          # Сбросить и пересоздать БД
    python populate_db_v3.py --chapter 1      # Только глава 1
    python populate_db_v3.py --history        # Показать историю изменений
    python populate_db_v3.py --history --run RUN_ID  # История конкретного прогона

v3.1 (2026-01-31): История изменений (error_history, sync_runs, diff-логика)
v3.0 (2026-01-31): Расширенные контексты, merge/split детекция
v2.0: Чтение из filtered.json
v1.0: Базовая версия
"""

VERSION = '3.2.0'  # v3.2: Унифицированный путь к БД из config.py

import sys
import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Set

# Добавляем путь к инструментам
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from morphology import get_lemma, get_pos
from filters.semantic_manager import SemanticManager, get_similarity
from filters.frequency_manager import FrequencyManager
from filters.comparison import levenshtein_distance
from filters.merge_split_detector import detect_all_patterns, MergeSplitPattern
from config import RESULTS_DIR, TESTS_DIR, FALSE_POSITIVES_DB
from error_context import (
    get_chapter_paths, text_to_words, extract_words_with_timing,
    get_context_window, build_word_positions, find_word_position,
    normalize_text_for_comparison
)
from version import PROJECT_VERSION, FILTER_ENGINE_VERSION

# Путь к БД — ЕДИНЫЙ источник из config.py
DB_PATH = FALSE_POSITIVES_DB

# Количество слов контекста с каждой стороны
CONTEXT_WINDOW = 10


class SimpleMorph:
    """Обёртка над функциями morphology"""
    def get_lemma(self, word: str) -> str:
        return get_lemma(word)

    def get_pos(self, word: str) -> str:
        return get_pos(word) or ''


class HistoryManager:
    """
    Управление историей изменений ошибок.

    Отслеживает:
    - created: новая ошибка появилась
    - deleted: ошибка исчезла (больше не находится)
    - filtered: ошибка стала фильтроваться
    - unfiltered: ошибка перестала фильтроваться
    - filter_reason_changed: изменилась причина фильтрации
    - golden_added: ошибка стала golden
    - golden_removed: ошибка перестала быть golden
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.run_id = str(uuid.uuid4())[:12]
        self.run_timestamp = datetime.now().isoformat()

    def start_run(self, chapters: List[int], reset: bool = False):
        """Начинает новый прогон синхронизации"""
        # Получаем количество ошибок до прогона
        cur = self.conn.execute('SELECT COUNT(*) FROM errors')
        errors_before = cur.fetchone()[0]

        self.conn.execute('''
            INSERT INTO sync_runs (
                run_id, run_timestamp, project_version, filter_version,
                chapters_processed, is_reset, total_errors_before
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.run_id,
            self.run_timestamp,
            PROJECT_VERSION,
            FILTER_ENGINE_VERSION,
            json.dumps(chapters),
            1 if reset else 0,
            errors_before,
        ))
        self.conn.commit()
        print(f"[HISTORY] Новый прогон: {self.run_id}")
        return self.run_id

    def finish_run(
        self,
        total_errors: int,
        golden_count: int,
        filtered_count: int,
        changes_count: int
    ):
        """Завершает прогон с итоговой статистикой"""
        # Считаем изменения по типам
        cur = self.conn.execute('''
            SELECT action, COUNT(*) FROM error_history
            WHERE run_id = ? GROUP BY action
        ''', (self.run_id,))
        action_counts = dict(cur.fetchall())

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
            total_errors,
            action_counts.get('created', 0),
            action_counts.get('deleted', 0),
            action_counts.get('filtered', 0),
            action_counts.get('unfiltered', 0),
            action_counts.get('golden_added', 0),
            action_counts.get('golden_removed', 0),
            self.run_id,
        ))
        self.conn.commit()
        print(f"[HISTORY] Прогон завершён. Изменений: {changes_count}")

    def get_existing_errors(self, chapter: int) -> Dict[str, Dict]:
        """Получает текущие ошибки главы для сравнения"""
        errors = {}
        cur = self.conn.execute('''
            SELECT error_id, wrong, correct, error_type, time_seconds,
                   is_filtered, filter_reason, is_golden, context
            FROM errors
            WHERE chapter = ?
        ''', (chapter,))

        for row in cur:
            key = self._make_error_key(row[1], row[2], row[3], row[4])
            errors[key] = {
                'error_id': row[0],
                'wrong': row[1],
                'correct': row[2],
                'error_type': row[3],
                'time_seconds': row[4],
                'is_filtered': row[5],
                'filter_reason': row[6],
                'is_golden': row[7],
                'context': row[8],
            }
        return errors

    def _make_error_key(
        self,
        wrong: str,
        correct: str,
        error_type: str,
        time_seconds: float
    ) -> str:
        """Создаёт ключ для идентификации ошибки"""
        # Ключ: wrong|correct|type|time_rounded
        time_key = round(time_seconds, 1) if time_seconds else 0
        return f"{wrong}|{correct}|{error_type}|{time_key}"

    def record_change(
        self,
        action: str,
        new_data: Dict,
        old_data: Optional[Dict] = None,
        chapter: int = 0
    ):
        """Записывает изменение в историю"""
        error_id = new_data.get('error_id', '') or (old_data or {}).get('error_id', '')

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
            error_id,
            chapter or new_data.get('chapter', 0),
            new_data.get('time_seconds', 0),
            new_data.get('wrong', ''),
            new_data.get('correct', ''),
            new_data.get('error_type', ''),
            action,
            (old_data or {}).get('is_filtered'),
            new_data.get('is_filtered'),
            (old_data or {}).get('filter_reason'),
            new_data.get('filter_reason'),
            (old_data or {}).get('is_golden'),
            new_data.get('is_golden'),
            new_data.get('context', ''),
            datetime.now().isoformat(),
        ))

    def compare_and_record(
        self,
        existing: Dict[str, Dict],
        new_errors: List[Dict],
        chapter: int
    ) -> Tuple[int, Set[str]]:
        """
        Сравнивает существующие и новые ошибки, записывает изменения.

        Returns:
            (changes_count, seen_keys)
        """
        changes = 0
        seen_keys = set()

        for new_data in new_errors:
            key = self._make_error_key(
                new_data.get('wrong', ''),
                new_data.get('correct', ''),
                new_data.get('error_type', ''),
                new_data.get('time_seconds', 0)
            )
            seen_keys.add(key)

            if key in existing:
                old_data = existing[key]

                # Сравниваем фильтрацию
                old_filtered = old_data.get('is_filtered', 0)
                new_filtered = new_data.get('is_filtered', 0)
                old_reason = old_data.get('filter_reason') or ''
                new_reason = new_data.get('filter_reason') or ''

                if old_filtered != new_filtered:
                    action = 'filtered' if new_filtered else 'unfiltered'
                    self.record_change(action, new_data, old_data, chapter)
                    changes += 1
                elif old_filtered and new_filtered and old_reason != new_reason:
                    self.record_change('filter_reason_changed', new_data, old_data, chapter)
                    changes += 1

                # Сравниваем golden
                old_golden = old_data.get('is_golden', 0)
                new_golden = new_data.get('is_golden', 0)

                if old_golden != new_golden:
                    action = 'golden_added' if new_golden else 'golden_removed'
                    self.record_change(action, new_data, old_data, chapter)
                    changes += 1
            else:
                # Новая ошибка
                self.record_change('created', new_data, chapter=chapter)
                changes += 1

        # Ошибки, которые исчезли
        for key, old_data in existing.items():
            if key not in seen_keys:
                self.record_change('deleted', old_data, chapter=chapter)
                changes += 1

        return changes, seen_keys


def seconds_to_label(seconds: float) -> str:
    """Конвертирует секунды в формат MM:SS"""
    if not seconds:
        return ""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def generate_error_id() -> str:
    """Генерирует уникальный ID ошибки"""
    return str(uuid.uuid4())[:8]


class DatabasePopulatorV3:
    """Заполняет БД ошибками с расширенным контекстом"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.morph = SimpleMorph()
        self.semantic = SemanticManager()
        self.frequency = FrequencyManager()
        self.golden_list = []
        self.history: Optional[HistoryManager] = None
        self.total_changes = 0

        # Кэш файлов по главам
        self.chapter_data: Dict[int, Dict[str, Any]] = {}

    def connect(self):
        """Подключение к БД"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    @staticmethod
    def normalize(word: str) -> str:
        """Нормализация слова для сопоставления"""
        return word.lower().replace('ё', 'е').strip()

    def load_golden(self):
        """Загрузка Golden стандарта"""
        self.golden_list = []
        for i in range(1, 6):
            path = TESTS_DIR / f'золотой_стандарт_глава{i}.json'
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for e in data.get('errors', []):
                        book_word = self.normalize(e.get('correct', ''))
                        said_word = self.normalize(e.get('wrong', ''))
                        time_sec = e.get('time_seconds', 0)
                        self.golden_list.append({
                            'chapter': i,
                            'book': book_word,
                            'said': said_word,
                            'time': time_sec
                        })
        print(f"[OK] Загружено {len(self.golden_list)} Golden ошибок")

    def is_golden(self, chapter: int, original: str, transcript: str, time_sec: float) -> bool:
        """Проверка на Golden ошибку по времени + совпадению слов"""
        TIME_TOLERANCE = 15

        book = self.normalize(original)
        said = self.normalize(transcript)

        for g in self.golden_list:
            if g['chapter'] != chapter:
                continue
            if abs(g['time'] - time_sec) > TIME_TOLERANCE:
                continue

            if book and g['book'] and book == g['book']:
                return True
            if said and g['said'] and said == g['said']:
                return True
            if g['book'] == '' and book == '' and g['said'] and said:
                if g['said'] == said:
                    return True
            if g['said'] == '' and said == '' and g['book'] and book:
                if g['book'] == book:
                    return True

        return False

    def load_chapter_data(self, chapter: int) -> Dict[str, Any]:
        """Загружает все данные главы для построения контекстов"""
        if chapter in self.chapter_data:
            return self.chapter_data[chapter]

        paths = get_chapter_paths(chapter)
        data = {
            'paths': paths,
            'original_words': [],
            'transcript_words': [],
            'normalized_words': [],
            'original_positions': [],
            'transcript_with_timing': [],
        }

        # Загружаем оригинал
        if paths['original_txt'].exists():
            text = paths['original_txt'].read_text(encoding='utf-8')
            data['original_words'] = text_to_words(text)
            data['original_positions'] = build_word_positions(text)

        # Загружаем нормализованный оригинал
        if paths['original_normalized'].exists():
            text = paths['original_normalized'].read_text(encoding='utf-8')
            data['normalized_words'] = text.split()

        # Загружаем транскрипцию с таймингами
        if paths['transcript_json'].exists():
            data['transcript_with_timing'] = extract_words_with_timing(paths['transcript_json'])
            data['transcript_words'] = [w['word'].lower() for w in data['transcript_with_timing']]

        self.chapter_data[chapter] = data
        return data

    def find_error_position_in_transcript(
        self,
        error: Dict,
        chapter_data: Dict
    ) -> Tuple[int, float, float]:
        """
        Находит позицию ошибки в транскрипции по времени.

        Returns:
            (position, time_start, time_end)
        """
        error_time = error.get('time', 0)
        transcript_words = chapter_data['transcript_with_timing']

        if not transcript_words:
            return -1, 0.0, 0.0

        # Ищем слово с ближайшим временем
        best_idx = -1
        best_diff = float('inf')

        for i, word_data in enumerate(transcript_words):
            diff = abs(word_data['start_time'] - error_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx >= 0:
            word_data = transcript_words[best_idx]
            return best_idx, word_data['start_time'], word_data['end_time']

        return -1, 0.0, 0.0

    def find_error_position_in_original(
        self,
        error: Dict,
        chapter_data: Dict
    ) -> int:
        """
        Находит позицию ошибки в оригинале по слову.

        Returns:
            position или -1
        """
        original_word = error.get('original', '') or error.get('correct', '')
        if not original_word:
            return -1

        original_positions = chapter_data['original_positions']
        if not original_positions:
            return -1

        # Ищем слово
        pos = find_word_position(original_word, original_positions)
        return pos if pos is not None else -1

    def get_context_from_words(
        self,
        words: List[str],
        center_idx: int,
        window: int = CONTEXT_WINDOW
    ) -> List[str]:
        """Извлекает контекст из списка слов"""
        if center_idx < 0 or not words:
            return []
        return get_context_window(words, center_idx, window)

    def get_window_timing(
        self,
        transcript_timing: List[Dict],
        center_idx: int,
        window: int = CONTEXT_WINDOW
    ) -> Tuple[float, float]:
        """
        Получает временные границы окна вокруг слова.

        Returns:
            (window_start, window_end)
        """
        if center_idx < 0 or not transcript_timing:
            return 0.0, 0.0

        start_idx = max(0, center_idx - window)
        end_idx = min(len(transcript_timing) - 1, center_idx + window)

        window_start = transcript_timing[start_idx]['start_time']
        window_end = transcript_timing[end_idx]['end_time']

        return window_start, window_end

    def process_error(
        self,
        error: Dict,
        chapter: int,
        error_index: int,
        chapter_data: Dict,
        pattern_info: Optional[Dict] = None
    ) -> Dict:
        """Обогащает ошибку расширенными метриками и контекстами"""

        # Базовые поля
        original_word = error.get('original', '').lower()
        transcript_word = error.get('transcript', '').lower()
        wrong = transcript_word
        correct = original_word
        error_type = error.get('type', 'substitution')
        time_sec = error.get('time', 0)

        # Генерируем error_id
        error_id = error.get('error_id', generate_error_id())

        # Находим позиции
        pos_transcript, time_start, time_end = self.find_error_position_in_transcript(
            error, chapter_data
        )
        pos_original = self.find_error_position_in_original(error, chapter_data)

        # Позиция в нормализованном = позиция в оригинале (те же слова)
        pos_normalized = pos_original

        # Получаем контексты
        context_transcript = self.get_context_from_words(
            chapter_data['transcript_words'], pos_transcript
        )
        context_original = self.get_context_from_words(
            chapter_data['original_words'], pos_original
        )
        context_normalized = self.get_context_from_words(
            chapter_data['normalized_words'], pos_normalized
        )

        # Временные окна
        window_start, window_end = self.get_window_timing(
            chapter_data['transcript_with_timing'], pos_transcript
        )

        # Расширенный контекст (±20 слов вместо ±10)
        context_start, context_end = self.get_window_timing(
            chapter_data['transcript_with_timing'], pos_transcript, window=20
        )

        # Морфология
        lemma_w = self.morph.get_lemma(wrong) if wrong else ''
        lemma_c = self.morph.get_lemma(correct) if correct else ''
        pos_w = self.morph.get_pos(wrong) if wrong else ''
        pos_c = self.morph.get_pos(correct) if correct else ''

        same_lemma = 1 if (lemma_w and lemma_c and lemma_w == lemma_c) else 0
        same_pos = 1 if (pos_w and pos_c and pos_w == pos_c) else 0

        # Семантика
        sem = 0.0
        if wrong and correct and error_type == 'substitution':
            sem = get_similarity(wrong, correct)

        # Частотность
        freq_w = self.frequency.get_frequency(wrong) if wrong else 0
        freq_c = self.frequency.get_frequency(correct) if correct else 0

        # Фонетика
        phon_sim = error.get('phonetic_similarity', error.get('similarity', 0))
        lev = levenshtein_distance(wrong, correct) if wrong and correct else 0

        # Фильтрация
        filter_reason = error.get('filter_reason')
        is_filtered = 1 if filter_reason else 0

        # Golden
        is_golden_flag = 1 if self.is_golden(chapter, correct, wrong, time_sec) else 0

        # Merge/Split паттерны
        linked_errors = []
        link_type = None
        merged_form = None
        split_parts = []

        if pattern_info:
            linked_errors = pattern_info.get('linked_errors', [])
            link_type = pattern_info.get('link_type')
            merged_form = pattern_info.get('merged_form')
            split_parts = pattern_info.get('split_parts', [])

        return {
            'error_id': error_id,
            'wrong': wrong,
            'correct': correct,
            'error_type': error_type,
            'chapter': chapter,
            'time_seconds': time_sec,
            'time_end_seconds': time_end,
            'time_label': seconds_to_label(time_sec),
            'window_start': window_start,
            'window_end': window_end,
            'context_start': context_start,
            'context_end': context_end,

            # Позиции
            'pos_transcript': pos_transcript,
            'pos_transcript_char': 0,  # TODO: вычислить
            'pos_normalized': pos_normalized,
            'pos_original': pos_original,
            'pos_original_char': 0,  # TODO: вычислить

            # Контексты (старые)
            'context': error.get('context', ''),
            'transcript_context': error.get('transcript_context', ''),

            # Контексты (новые, как JSON)
            'context_transcript': json.dumps(context_transcript, ensure_ascii=False),
            'context_normalized': json.dumps(context_normalized, ensure_ascii=False),
            'context_original': json.dumps(context_original, ensure_ascii=False),
            'context_aligned': '',  # TODO: заполнить из alignment

            # Связи
            'linked_errors': json.dumps(linked_errors, ensure_ascii=False) if linked_errors else None,
            'link_type': link_type,
            'merged_form': merged_form,
            'split_parts': json.dumps(split_parts, ensure_ascii=False) if split_parts else None,

            # Сегменты
            'segment_id': -1,
            'is_boundary': 0,

            # Golden и фильтрация
            'is_golden': is_golden_flag,
            'is_filtered': is_filtered,
            'filter_reason': filter_reason,

            # Морфология
            'lemma_wrong': lemma_w,
            'lemma_correct': lemma_c,
            'pos_wrong': pos_w,
            'pos_correct': pos_c,
            'same_lemma': same_lemma,
            'same_pos': same_pos,

            # Семантика и частотность
            'semantic_similarity': sem,
            'frequency_wrong': freq_w,
            'frequency_correct': freq_c,

            # Фонетика
            'phonetic_similarity': phon_sim,
            'levenshtein': lev,

            # Метаданные
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'schema_version': 2,
        }

    def insert_error(self, data: Dict):
        """Вставка ошибки в БД"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO errors (
                    error_id, wrong, correct, error_type, chapter,
                    time_seconds, time_end_seconds, time_label,
                    window_start, window_end, context_start, context_end,
                    pos_transcript, pos_transcript_char, pos_normalized, pos_original, pos_original_char,
                    context, transcript_context,
                    context_transcript, context_normalized, context_original, context_aligned,
                    linked_errors, link_type, merged_form, split_parts,
                    segment_id, is_boundary,
                    is_golden, is_filtered, filter_reason,
                    lemma_wrong, lemma_correct, pos_wrong, pos_correct, same_lemma, same_pos,
                    semantic_similarity, frequency_wrong, frequency_correct,
                    phonetic_similarity, levenshtein,
                    created_at, updated_at, schema_version
                ) VALUES (
                    :error_id, :wrong, :correct, :error_type, :chapter,
                    :time_seconds, :time_end_seconds, :time_label,
                    :window_start, :window_end, :context_start, :context_end,
                    :pos_transcript, :pos_transcript_char, :pos_normalized, :pos_original, :pos_original_char,
                    :context, :transcript_context,
                    :context_transcript, :context_normalized, :context_original, :context_aligned,
                    :linked_errors, :link_type, :merged_form, :split_parts,
                    :segment_id, :is_boundary,
                    :is_golden, :is_filtered, :filter_reason,
                    :lemma_wrong, :lemma_correct, :pos_wrong, :pos_correct, :same_lemma, :same_pos,
                    :semantic_similarity, :frequency_wrong, :frequency_correct,
                    :phonetic_similarity, :levenshtein,
                    :created_at, :updated_at, :schema_version
                )
            ''', data)
        except Exception as e:
            print(f"[WARN] Ошибка вставки: {e}")

    def insert_error_link(self, pattern: MergeSplitPattern, error_ids: List[str]):
        """Вставка связи между ошибками"""
        if len(error_ids) < 2:
            return

        link_id = str(uuid.uuid4())[:8]

        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO error_links (
                    link_id, error1_id, error2_id, link_type, pattern,
                    original_parts, merged_form, confidence, chapter,
                    time_start, time_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                link_id,
                error_ids[0],
                error_ids[1],
                pattern.pattern_type + '_artifact',
                pattern.pattern_str,
                json.dumps(pattern.original_parts, ensure_ascii=False),
                pattern.merged_form,
                pattern.confidence,
                pattern.chapter,
                pattern.time_start,
                pattern.time_end,
                datetime.now().isoformat(),
            ))
        except Exception as e:
            print(f"[WARN] Ошибка вставки связи: {e}")

    def populate_chapter(
        self,
        chapter: int,
        reset: bool = False
    ) -> Dict[str, int]:
        """Заполняет БД ошибками одной главы с отслеживанием изменений"""
        stats = {
            'total': 0,
            'golden': 0,
            'filtered': 0,
            'merge_patterns': 0,
            'split_patterns': 0,
        }

        # Загружаем данные главы
        chapter_data = self.load_chapter_data(chapter)
        paths = chapter_data['paths']

        # Читаем compared.json для детекции паттернов
        compared_path = paths['compared_json']
        filtered_path = paths['filtered_json']

        if not filtered_path.exists():
            print(f"[SKIP] Нет файла: {filtered_path}")
            return stats

        # Получаем существующие ошибки для diff (если не reset)
        existing_errors = {}
        if not reset and self.history:
            existing_errors = self.history.get_existing_errors(chapter)

        # Загружаем compared для детекции паттернов
        compared_errors = []
        if compared_path.exists():
            with open(compared_path, 'r', encoding='utf-8') as f:
                compared_data = json.load(f)
                compared_errors = compared_data.get('errors', [])

        # Детектируем паттерны
        merge_patterns, split_patterns = detect_all_patterns(compared_errors, chapter=chapter)
        stats['merge_patterns'] = len(merge_patterns)
        stats['split_patterns'] = len(split_patterns)

        # Строим маппинг индексов ошибок к паттернам
        index_to_pattern: Dict[int, Dict] = {}
        for pattern in merge_patterns + split_patterns:
            for idx in pattern.error_indices:
                index_to_pattern[idx] = {
                    'pattern': pattern,
                    'link_type': pattern.pattern_type + '_artifact',
                    'merged_form': pattern.merged_form,
                    'split_parts': pattern.transcript_parts if pattern.pattern_type == 'split' else [],
                    'linked_indices': pattern.error_indices,
                }

        # Загружаем filtered.json
        with open(filtered_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        remaining = data.get('errors', [])
        filtered_list = data.get('filtered_errors_detail', [])
        all_errors = remaining + filtered_list

        print(f"\n[Глава {chapter}] {len(all_errors)} ошибок, {len(merge_patterns)} merge, {len(split_patterns)} split...")

        # Обрабатываем ошибки и собираем для diff
        error_id_map: Dict[int, str] = {}  # Индекс → error_id
        processed_errors: List[Dict] = []  # Для diff

        for idx, error in enumerate(all_errors):
            # Проверяем есть ли паттерн для этой ошибки
            pattern_info = None
            if idx in index_to_pattern:
                pi = index_to_pattern[idx]
                pattern_info = {
                    'link_type': pi['link_type'],
                    'merged_form': pi['merged_form'],
                    'split_parts': pi['split_parts'],
                    'linked_errors': [],  # Заполним позже
                }

            enriched = self.process_error(error, chapter, idx, chapter_data, pattern_info)
            error_id_map[idx] = enriched['error_id']
            processed_errors.append(enriched)

            self.insert_error(enriched)
            stats['total'] += 1

            if enriched['is_golden']:
                stats['golden'] += 1
            if enriched['is_filtered']:
                stats['filtered'] += 1

            if (idx + 1) % 100 == 0:
                print(f"  ...обработано {idx+1}/{len(all_errors)}")

        # Записываем изменения в историю
        if self.history and not reset:
            changes, _ = self.history.compare_and_record(
                existing_errors, processed_errors, chapter
            )
            self.total_changes += changes
            if changes > 0:
                print(f"  [HISTORY] {changes} изменений записано")

        # Обновляем linked_errors с реальными error_id
        for pattern in merge_patterns + split_patterns:
            linked_ids = [error_id_map.get(i) for i in pattern.error_indices if i in error_id_map]
            if len(linked_ids) >= 2:
                self.insert_error_link(pattern, linked_ids)

                # Обновляем linked_errors в записях
                for idx in pattern.error_indices:
                    if idx in error_id_map:
                        other_ids = [error_id_map.get(i) for i in pattern.error_indices if i != idx and i in error_id_map]
                        if other_ids:
                            self.conn.execute(
                                'UPDATE errors SET linked_errors = ? WHERE error_id = ?',
                                (json.dumps(other_ids, ensure_ascii=False), error_id_map[idx])
                            )

        self.conn.commit()
        return stats

    def populate(self, reset: bool = False, chapters: List[int] = None):
        """Основной метод заполнения БД с отслеживанием истории"""
        self.connect()
        self.load_golden()

        if chapters is None:
            chapters = [1, 2, 3, 4, 5]

        # Инициализируем историю
        self.history = HistoryManager(self.conn)
        self.history.start_run(chapters, reset)
        self.total_changes = 0

        if reset:
            # Очищаем таблицы
            self.conn.execute('DELETE FROM errors')
            self.conn.execute('DELETE FROM error_links')
            self.conn.execute('DELETE FROM alignment_segments')
            self.conn.commit()
            print("[OK] Таблицы очищены")

        total_stats = {
            'total': 0,
            'golden': 0,
            'filtered': 0,
            'merge_patterns': 0,
            'split_patterns': 0,
        }

        for chapter in chapters:
            stats = self.populate_chapter(chapter, reset=reset)
            for key in total_stats:
                total_stats[key] += stats[key]

        # Завершаем историю
        self.history.finish_run(
            total_stats['total'],
            total_stats['golden'],
            total_stats['filtered'],
            self.total_changes
        )

        print(f"\n{'='*60}")
        print(f"[DONE] Итого:")
        print(f"  Ошибок: {total_stats['total']}")
        print(f"  Golden: {total_stats['golden']}")
        print(f"  Отфильтровано: {total_stats['filtered']}")
        print(f"  Merge patterns: {total_stats['merge_patterns']}")
        print(f"  Split patterns: {total_stats['split_patterns']}")
        print(f"  Linked ошибок: {(total_stats['merge_patterns'] + total_stats['split_patterns']) * 2}")
        print(f"  Изменений в истории: {self.total_changes}")

    def show_stats(self):
        """Показать статистику БД"""
        self.connect()

        print(f"\n{'='*60}")
        print(f"СТАТИСТИКА БД v3.0")
        print(f"{'='*60}")

        # Общая статистика
        cur = self.conn.execute('SELECT COUNT(*) FROM errors')
        total = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE is_golden = 1')
        golden = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE is_filtered = 1')
        filtered = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE link_type IS NOT NULL')
        linked = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM error_links')
        links = cur.fetchone()[0]

        print(f"Всего ошибок: {total}")
        print(f"Golden: {golden}")
        print(f"Отфильтровано: {filtered}")
        print(f"С контекстами: {total}")
        print(f"Linked ошибок: {linked}")
        print(f"Связей (error_links): {links}")

        # По типам связей
        print(f"\n--- По типам связей ---")
        cur = self.conn.execute('''
            SELECT link_type, COUNT(*) as cnt
            FROM error_links
            GROUP BY link_type
            ORDER BY cnt DESC
        ''')
        for row in cur:
            print(f"  {row[0]}: {row[1]}")

        # Пример связи
        print(f"\n--- Примеры связей ---")
        cur = self.conn.execute('''
            SELECT pattern, link_type, chapter, time_start
            FROM error_links
            ORDER BY chapter, time_start
            LIMIT 10
        ''')
        for row in cur:
            print(f"  {row[0]} ({row[1]}) @ глава {row[2]}, {row[3]:.1f}s")

    def show_history(self, run_id: Optional[str] = None, limit: int = 50):
        """Показать историю изменений"""
        self.connect()

        print(f"\n{'='*60}")
        print(f"ИСТОРИЯ ИЗМЕНЕНИЙ v3.1")
        print(f"{'='*60}")

        # Статистика по прогонам
        cur = self.conn.execute('''
            SELECT run_id, run_timestamp, project_version, filter_version,
                   total_errors_before, total_errors_after,
                   errors_created, errors_deleted, errors_filtered, errors_unfiltered,
                   golden_added, golden_removed, is_reset, run_finished
            FROM sync_runs
            ORDER BY run_timestamp DESC
            LIMIT 10
        ''')
        runs = cur.fetchall()

        if not runs:
            print("\n[INFO] История прогонов пуста")
            return

        print(f"\n--- Последние прогоны ({len(runs)}) ---")
        for row in runs:
            rid, ts, pv, fv, before, after, created, deleted, filt, unfilt, g_add, g_rem, reset, finished = row
            status = "done" if finished else "running"
            reset_flag = " [RESET]" if reset else ""
            total_changes = (created or 0) + (deleted or 0) + (filt or 0) + (unfilt or 0) + (g_add or 0) + (g_rem or 0)
            print(f"  {rid[:8]}  {ts[:19]}  v{pv}  filter v{fv}{reset_flag}")
            print(f"           errors: {before or '?'}→{after or '?'}  changes={total_changes} [{status}]")
            if total_changes > 0:
                details = []
                if created: details.append(f"+{created}")
                if deleted: details.append(f"-{deleted}")
                if filt: details.append(f"filt:{filt}")
                if unfilt: details.append(f"unfilt:{unfilt}")
                if g_add: details.append(f"g+:{g_add}")
                if g_rem: details.append(f"g-:{g_rem}")
                print(f"           {', '.join(details)}")

        # Если указан run_id — показываем детали
        if run_id:
            self._show_run_details(run_id)
        else:
            # Показываем последние изменения
            self._show_recent_changes(limit)

    def _show_run_details(self, run_id: str):
        """Детали конкретного прогона"""
        print(f"\n--- Детали прогона {run_id} ---")

        cur = self.conn.execute('''
            SELECT action, COUNT(*) as cnt
            FROM error_history
            WHERE run_id LIKE ?
            GROUP BY action
            ORDER BY cnt DESC
        ''', (f'{run_id}%',))

        actions = cur.fetchall()
        if not actions:
            print(f"  [WARN] Прогон не найден или нет изменений")
            return

        print(f"\nДействия:")
        for action, cnt in actions:
            print(f"  {action}: {cnt}")

        # Примеры изменений
        print(f"\nПримеры изменений:")
        cur = self.conn.execute('''
            SELECT action, wrong, correct, chapter, time_seconds,
                   old_filter_reason, new_filter_reason
            FROM error_history
            WHERE run_id LIKE ?
            ORDER BY chapter, time_seconds
            LIMIT 20
        ''', (f'{run_id}%',))

        for row in cur:
            action, wrong, correct, ch, t, old_r, new_r = row
            time_str = f"{int(t)//60}:{int(t)%60:02d}" if t else "?"

            if action == 'filtered':
                print(f"  [FILTERED] {wrong}→{correct} @{ch}/{time_str}: {new_r}")
            elif action == 'unfiltered':
                print(f"  [UNFILTERED] {wrong}→{correct} @{ch}/{time_str}: было {old_r}")
            elif action == 'created':
                print(f"  [NEW] {wrong}→{correct} @{ch}/{time_str}")
            elif action == 'deleted':
                print(f"  [DELETED] {wrong}→{correct} @{ch}/{time_str}")
            elif action == 'golden_added':
                print(f"  [GOLDEN+] {wrong}→{correct} @{ch}/{time_str}")
            elif action == 'golden_removed':
                print(f"  [GOLDEN-] {wrong}→{correct} @{ch}/{time_str}")
            else:
                print(f"  [{action.upper()}] {wrong}→{correct} @{ch}/{time_str}")

    def _show_recent_changes(self, limit: int):
        """Показать последние изменения"""
        print(f"\n--- Последние {limit} изменений ---")

        cur = self.conn.execute('''
            SELECT action, wrong, correct, chapter, time_seconds,
                   old_filter_reason, new_filter_reason, run_id, run_timestamp
            FROM error_history
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        changes = cur.fetchall()
        if not changes:
            print("  [INFO] Нет записей в истории")
            return

        for row in changes:
            action, wrong, correct, ch, t, old_r, new_r, rid, ts = row
            time_str = f"{int(t)//60}:{int(t)%60:02d}" if t else "?"

            if action == 'filtered':
                print(f"  [{rid[:6]}] FILTERED {wrong}→{correct} @{ch}/{time_str}: {new_r}")
            elif action == 'unfiltered':
                print(f"  [{rid[:6]}] UNFILTERED {wrong}→{correct} @{ch}/{time_str}")
            elif action == 'created':
                print(f"  [{rid[:6]}] NEW {wrong}→{correct} @{ch}/{time_str}")
            elif action == 'deleted':
                print(f"  [{rid[:6]}] DELETED {wrong}→{correct} @{ch}/{time_str}")
            else:
                print(f"  [{rid[:6]}] {action.upper()} {wrong}→{correct} @{ch}/{time_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Заполнение БД ошибок v3.1 с историей')
    parser.add_argument('--reset', action='store_true', help='Сбросить и пересоздать БД')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    parser.add_argument('--history', action='store_true', help='Показать историю изменений')
    parser.add_argument('--run', type=str, help='ID прогона для детальной истории')
    parser.add_argument('--chapter', type=int, help='Только указанная глава')
    parser.add_argument('--limit', type=int, default=50, help='Лимит записей истории')
    args = parser.parse_args()

    print(f"Populate DB v{VERSION}")
    print("=" * 60)

    pop = DatabasePopulatorV3()

    if args.history:
        pop.show_history(run_id=args.run, limit=args.limit)
    elif args.stats:
        pop.show_stats()
    else:
        chapters = [args.chapter] if args.chapter else None
        pop.populate(reset=args.reset, chapters=chapters)
        pop.show_stats()


if __name__ == '__main__':
    main()
