#!/usr/bin/env python3
"""
Populate False Positives Database v1.0

Заполняет БД всеми ошибками из compared.json файлов,
прогоняет через фильтры и добавляет метрики:
- Морфология: lemma, POS, same_lemma
- Семантика: semantic_similarity
- Частотность: frequency_wrong, frequency_correct
- Результат фильтрации: is_filtered, filter_reason

Использование:
    python populate_db.py                  # Заполнить БД
    python populate_db.py --stats          # Показать статистику
    python populate_db.py --reset          # Сбросить и пересоздать БД
"""

VERSION = '1.0.0'

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Добавляем путь к инструментам
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from morphology import get_lemma, get_pos
from filters.semantic_manager import SemanticManager, get_similarity
from filters.frequency_manager import FrequencyManager
from filters.engine import should_filter_error
from config import RESULTS_DIR, TESTS_DIR, DICTIONARIES_DIR


class SimpleMorph:
    """Обёртка над функциями morphology"""
    def get_lemma(self, word: str) -> str:
        return get_lemma(word)

    def get_pos(self, word: str) -> str:
        return get_pos(word) or ''

# Путь к БД
DB_PATH = DICTIONARIES_DIR / 'false_positives.db'

# Схема БД v1
SCHEMA = '''
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Базовые поля
    wrong TEXT NOT NULL,
    correct TEXT NOT NULL,
    error_type TEXT NOT NULL,              -- substitution, insertion, deletion
    chapter INTEGER NOT NULL,              -- 1-4
    time_seconds REAL,
    time_label TEXT,                       -- "12:34"
    context TEXT,

    -- Golden статус
    is_golden INTEGER DEFAULT 0,           -- 1 если реальная ошибка чтеца

    -- Результат фильтрации
    is_filtered INTEGER DEFAULT 0,         -- 1 если отфильтровано
    filter_reason TEXT,                    -- название фильтра

    -- Морфология
    lemma_wrong TEXT,
    lemma_correct TEXT,
    pos_wrong TEXT,
    pos_correct TEXT,
    same_lemma INTEGER DEFAULT 0,          -- 1 если одинаковая лемма
    same_pos INTEGER DEFAULT 0,            -- 1 если одинаковая часть речи

    -- Семантика
    semantic_similarity REAL DEFAULT 0,    -- косинусное сходство 0-1

    -- Частотность (ipm из НКРЯ)
    frequency_wrong REAL DEFAULT 0,
    frequency_correct REAL DEFAULT 0,

    -- Фонетика
    phonetic_similarity REAL DEFAULT 0,    -- из compared.json
    levenshtein INTEGER DEFAULT 0,         -- расстояние Левенштейна

    -- Метаданные
    created_at TEXT NOT NULL,

    -- Уникальность по главе + время + слова
    UNIQUE(chapter, time_seconds, wrong, correct)
);

CREATE INDEX IF NOT EXISTS idx_errors_golden ON errors(is_golden);
CREATE INDEX IF NOT EXISTS idx_errors_filtered ON errors(is_filtered);
CREATE INDEX IF NOT EXISTS idx_errors_chapter ON errors(chapter);
CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type);
CREATE INDEX IF NOT EXISTS idx_errors_same_lemma ON errors(same_lemma);
CREATE INDEX IF NOT EXISTS idx_errors_semantic ON errors(semantic_similarity);
'''


def levenshtein(s1: str, s2: str) -> int:
    """Расстояние Левенштейна"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def seconds_to_label(seconds: float) -> str:
    """Конвертирует секунды в формат MM:SS"""
    if not seconds:
        return ""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class DatabasePopulator:
    """Заполняет БД ошибками с метриками"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.morph = SimpleMorph()
        self.semantic = SemanticManager()
        self.frequency = FrequencyManager()
        self.golden_set = set()

    def connect(self):
        """Подключение к БД"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def create_schema(self, reset: bool = False):
        """Создание схемы БД"""
        if reset and self.db_path.exists():
            self.db_path.unlink()
            self.connect()

        self.conn.executescript(SCHEMA)
        self.conn.commit()
        print(f"[OK] Схема БД создана: {self.db_path}")

    @staticmethod
    def normalize(word: str) -> str:
        """Нормализация слова для сопоставления"""
        return word.lower().replace('ё', 'е').strip()

    def load_golden(self):
        """Загрузка Golden стандарта.

        Golden файлы содержат:
        - wrong = что сказал чтец (или что распознал Яндекс)
        - correct = что в книге
        - time_seconds = время ошибки

        Сопоставление по времени (±15 сек) + совпадению хотя бы одного слова.
        """
        self.golden_list = []
        for i in range(1, 5):
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
        """Проверка на Golden ошибку по времени + совпадению слов.

        Логика:
        1. Ищем golden ошибки в пределах ±15 секунд
        2. Проверяем совпадение хотя бы одного слова (book или said)

        Это работает для всех типов ошибок:
        - substitution: book (original) совпадает
        - deletion: book (original) совпадает, said пустой
        - insertion: said (transcript) совпадает, book пустой
        """
        TIME_TOLERANCE = 15

        book = self.normalize(original)
        said = self.normalize(transcript)

        for g in self.golden_list:
            if g['chapter'] != chapter:
                continue
            if abs(g['time'] - time_sec) > TIME_TOLERANCE:
                continue

            # Проверяем совпадение слов
            # 1. Книжное слово (original) совпадает
            if book and g['book'] and book == g['book']:
                return True

            # 2. Сказанное слово (transcript) совпадает
            if said and g['said'] and said == g['said']:
                return True

            # 3. Для insertion/deletion с пустыми словами
            if g['book'] == '' and book == '' and g['said'] and said:
                # Оба insertion — проверяем сказанное
                if g['said'] == said:
                    return True
            if g['said'] == '' and said == '' and g['book'] and book:
                # Оба deletion — проверяем книжное
                if g['book'] == book:
                    return True

        return False

    def process_error(self, error: Dict, chapter: int) -> Dict:
        """Обогащает ошибку метриками

        В compared.json:
        - original = слово из книги
        - transcript = слово распознанное Яндексом

        В БД храним:
        - wrong = transcript (что распознано/сказано)
        - correct = original (что в книге)
        """
        # original = книга, transcript = распознано
        original_word = error.get('original', '').lower()
        transcript_word = error.get('transcript', '').lower()

        # В БД: wrong = сказано, correct = книга
        wrong = transcript_word
        correct = original_word
        error_type = error.get('type', 'substitution')
        time_sec = error.get('time', 0)

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
        lev = levenshtein(wrong, correct) if wrong and correct else 0

        # Фильтрация — читаем из error (filter_reason)
        # filter_errors() добавляет filter_reason к отфильтрованным ошибкам
        filter_reason = error.get('filter_reason')
        is_filtered = 1 if filter_reason else 0

        # Golden — проверяем по (глава, книга, сказано, время)
        is_golden = 1 if self.is_golden(chapter, correct, wrong, time_sec) else 0

        return {
            'wrong': wrong,
            'correct': correct,
            'error_type': error_type,
            'chapter': chapter,
            'time_seconds': time_sec,
            'time_label': seconds_to_label(time_sec),
            'context': error.get('context', ''),
            'is_golden': is_golden,
            'is_filtered': is_filtered,
            'filter_reason': filter_reason,
            'lemma_wrong': lemma_w,
            'lemma_correct': lemma_c,
            'pos_wrong': pos_w,
            'pos_correct': pos_c,
            'same_lemma': same_lemma,
            'same_pos': same_pos,
            'semantic_similarity': sem,
            'frequency_wrong': freq_w,
            'frequency_correct': freq_c,
            'phonetic_similarity': phon_sim,
            'levenshtein': lev,
            'created_at': datetime.now().isoformat()
        }

    def insert_error(self, data: Dict):
        """Вставка ошибки в БД"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO errors (
                    wrong, correct, error_type, chapter, time_seconds, time_label, context,
                    is_golden, is_filtered, filter_reason,
                    lemma_wrong, lemma_correct, pos_wrong, pos_correct, same_lemma, same_pos,
                    semantic_similarity, frequency_wrong, frequency_correct,
                    phonetic_similarity, levenshtein, created_at
                ) VALUES (
                    :wrong, :correct, :error_type, :chapter, :time_seconds, :time_label, :context,
                    :is_golden, :is_filtered, :filter_reason,
                    :lemma_wrong, :lemma_correct, :pos_wrong, :pos_correct, :same_lemma, :same_pos,
                    :semantic_similarity, :frequency_wrong, :frequency_correct,
                    :phonetic_similarity, :levenshtein, :created_at
                )
            ''', data)
        except Exception as e:
            print(f"[WARN] Ошибка вставки: {e}")

    def populate(self, reset: bool = False):
        """Основной метод заполнения БД.

        ВАЖНО: Читаем compared.json и прогоняем через filter_errors(),
        чтобы гарантировать консистентность с filtered.json.
        Это единственный источник правды о фильтрации.
        """
        from filters.engine import filter_errors

        self.connect()
        self.create_schema(reset=reset)
        self.load_golden()

        total = 0
        golden_count = 0
        filtered_count = 0

        # Обрабатываем каждую главу
        for i in range(1, 5):
            chapter_dir = RESULTS_DIR / f'0{i}'
            compared_path = chapter_dir / f'0{i}_compared.json'

            if not compared_path.exists():
                print(f"[SKIP] Нет файла: {compared_path}")
                continue

            with open(compared_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            errors = data.get('errors', [])

            # Прогоняем через ТОТЖЕ filter_errors что и pipeline!
            remaining, filtered_list, stats = filter_errors(errors)

            # remaining — ошибки БЕЗ флага filtered
            # filtered_list — ошибки С флагом filtered и filter_reason

            all_errors = remaining + filtered_list
            print(f"\n[Глава {i}] Обрабатываю {len(all_errors)} ошибок (filtered: {len(filtered_list)}, remaining: {len(remaining)})...")

            for j, error in enumerate(all_errors):
                enriched = self.process_error(error, chapter=i)
                self.insert_error(enriched)
                total += 1
                if enriched['is_golden']:
                    golden_count += 1
                if enriched['is_filtered']:
                    filtered_count += 1

                if (j + 1) % 100 == 0:
                    print(f"  ...обработано {j+1}/{len(all_errors)}")

        self.conn.commit()

        print(f"\n{'='*50}")
        print(f"[DONE] Добавлено {total} ошибок в БД")
        print(f"  - Golden: {golden_count}")
        print(f"  - Отфильтровано: {filtered_count}")
        print(f"  - Осталось FP: {total - golden_count - filtered_count}")

    def show_stats(self):
        """Показать статистику БД"""
        self.connect()

        # Общая статистика
        cur = self.conn.execute('SELECT COUNT(*) FROM errors')
        total = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE is_golden = 1')
        golden = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE is_filtered = 1')
        filtered = cur.fetchone()[0]

        cur = self.conn.execute('SELECT COUNT(*) FROM errors WHERE is_golden = 0 AND is_filtered = 0')
        fp_remaining = cur.fetchone()[0]

        print(f"\n{'='*50}")
        print(f"СТАТИСТИКА БАЗЫ ДАННЫХ")
        print(f"{'='*50}")
        print(f"Всего ошибок: {total}")
        print(f"Golden (реальные ошибки): {golden}")
        print(f"Отфильтровано: {filtered}")
        print(f"Осталось FP: {fp_remaining}")

        # По типам
        print(f"\n--- По типам ---")
        cur = self.conn.execute('''
            SELECT error_type, COUNT(*) as cnt
            FROM errors
            GROUP BY error_type
            ORDER BY cnt DESC
        ''')
        for row in cur:
            print(f"  {row['error_type']}: {row['cnt']}")

        # По фильтрам
        print(f"\n--- Топ-10 фильтров ---")
        cur = self.conn.execute('''
            SELECT filter_reason, COUNT(*) as cnt
            FROM errors
            WHERE is_filtered = 1 AND filter_reason IS NOT NULL
            GROUP BY filter_reason
            ORDER BY cnt DESC
            LIMIT 10
        ''')
        for row in cur:
            print(f"  {row['filter_reason']}: {row['cnt']}")

        # Категории нефильтруемых FP
        print(f"\n--- Нефильтруемые FP по категориям ---")

        # same_lemma + high_sem
        cur = self.conn.execute('''
            SELECT COUNT(*) FROM errors
            WHERE is_golden = 0 AND is_filtered = 0
            AND same_lemma = 1 AND semantic_similarity >= 0.5
        ''')
        cat1 = cur.fetchone()[0]
        print(f"  same_lemma + sem>=0.5: {cat1}")

        # same_lemma + low_sem
        cur = self.conn.execute('''
            SELECT COUNT(*) FROM errors
            WHERE is_golden = 0 AND is_filtered = 0
            AND same_lemma = 1 AND semantic_similarity < 0.5
        ''')
        cat2 = cur.fetchone()[0]
        print(f"  same_lemma + sem<0.5: {cat2}")

        # diff_lemma + high_sem
        cur = self.conn.execute('''
            SELECT COUNT(*) FROM errors
            WHERE is_golden = 0 AND is_filtered = 0
            AND same_lemma = 0 AND semantic_similarity >= 0.5
        ''')
        cat3 = cur.fetchone()[0]
        print(f"  diff_lemma + sem>=0.5: {cat3}")

        # diff_lemma + no_sem (not in vocab)
        cur = self.conn.execute('''
            SELECT COUNT(*) FROM errors
            WHERE is_golden = 0 AND is_filtered = 0
            AND same_lemma = 0 AND semantic_similarity = 0
            AND error_type = 'substitution'
        ''')
        cat4 = cur.fetchone()[0]
        print(f"  diff_lemma + sem=0 (не в словаре): {cat4}")

        # insertions/deletions
        cur = self.conn.execute('''
            SELECT COUNT(*) FROM errors
            WHERE is_golden = 0 AND is_filtered = 0
            AND error_type != 'substitution'
        ''')
        cat5 = cur.fetchone()[0]
        print(f"  insertions/deletions: {cat5}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Заполнение БД ошибок')
    parser.add_argument('--reset', action='store_true', help='Сбросить и пересоздать БД')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    args = parser.parse_args()

    pop = DatabasePopulator()

    if args.stats:
        pop.show_stats()
    else:
        pop.populate(reset=args.reset)
        pop.show_stats()


if __name__ == '__main__':
    main()
