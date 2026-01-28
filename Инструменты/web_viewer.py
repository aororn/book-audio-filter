#!/usr/bin/env python3
"""
Web Viewer v3.0 - Веб-интерфейс для просмотра результатов проверки транскрипции

Функции:
- Просмотр результатов в браузере
- Интерактивный аудиоплеер с навигацией по таймкодам
- Фильтрация ошибок по типу
- Экспорт в различные форматы
- Автоопределение связанных файлов по конвенции именования

Использование:
    python web_viewer.py 01_filtered.json
    python web_viewer.py 01_filtered.json --audio аудио.mp3
    python web_viewer.py --port 8080
"""

import argparse
import html
import json
import os
import re
import sys
import webbrowser
import threading
import logging
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import urllib.parse

# Импорт централизованной конфигурации
try:
    from config import (
        READER_DIR, RESULTS_DIR, TEMP_DIR, CHAPTERS_DIR, AUDIO_DIR,
        TRANSCRIPTIONS_DIR, FileNaming, ensure_dirs_exist
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    AUDIO_DIR = None
    TRANSCRIPTIONS_DIR = None

# Импорт модуля DOCX-экспорта (выделен из web_viewer.py v5.1.0)
try:
    from docx_export import generate_reader_docx as _generate_reader_docx_module
    HAS_DOCX_EXPORT = True
except ImportError:
    HAS_DOCX_EXPORT = False

# === СИСТЕМА ЛОГИРОВАНИЯ ===
# Логи сохраняются в папку Темп для отладки проблем без скриншотов

LOG_DIR = None
DEBUG_STATE = {
    'errors': [],
    'original_loaded': False,
    'transcript_loaded': False,
    'enrichment_log': [],
    'last_request': None,
}

# === ЕДИНЫЕ ПРАВИЛА ДЛЯ РАБОТЫ С КОНТЕКСТОМ ===
# Эти константы определяют поведение всех функций обогащения и отображения контекста
#
# ГЛАВНЫЙ ПРИНЦИП: Контекст из JSON (smart_compare) считается правильным.
# Обогащение добавляет пунктуацию, но НЕ ДОЛЖНО менять место в тексте.
#
# Правила обогащения:
# 1. Короткие слова (≤4 символов) — контекст НЕ перезаписывается
# 2. Если контекст уже содержит слово ошибки — контекст НЕ перезаписывается
# 3. Для insertion: если контекст уже есть (>50 символов) — НЕ перезаписывается
# 4. Обогащение только добавляет пунктуацию из оригинала

# Минимальная длина слова для поиска в оригинале.
# Слова короче этого порога НЕ используются для поиска позиции в тексте,
# так как они встречаются слишком часто (и, а, в, так, что, это, его, её).
SHORT_WORD_THRESHOLD = 4

# Минимальная длина слова для добавления в приоритетный список ключевых слов
PRIORITY_WORD_MIN_LENGTH = 5

# Минимальная длина ключевых слов для поиска контекста
KEY_WORD_MIN_LENGTH_PRIMARY = 6  # Первый приоритет
KEY_WORD_MIN_LENGTH_FALLBACK = 4  # Если нет длинных слов

# Минимальная длина существующего контекста для пропуска обогащения (insertion)
MIN_EXISTING_CONTEXT_LENGTH = 50


def normalize_for_search(text: str) -> str:
    """
    Нормализует текст для поиска: lowercase + ё→е.
    Используется везде где нужно сравнивать слова.
    """
    return text.lower().replace('ё', 'е')


def is_short_word(word: str) -> bool:
    """
    Проверяет, является ли слово "коротким" для целей поиска контекста.
    Короткие слова встречаются слишком часто и дают ложные совпадения.
    """
    return len(word) <= SHORT_WORD_THRESHOLD


def get_key_words_from_context(context: str, max_words: int = 7) -> list:
    """
    Извлекает ключевые (длинные, уникальные) слова из контекста для поиска.
    Возвращает список слов, отсортированных по убыванию длины.
    """
    words = context.split()
    # Сначала берём длинные слова (≥6 символов)
    key_words = [w for w in words if len(w) >= KEY_WORD_MIN_LENGTH_PRIMARY][:max_words]
    # Если нет длинных, берём средние (≥4 символа)
    if not key_words:
        key_words = [w for w in words if len(w) >= KEY_WORD_MIN_LENGTH_FALLBACK][:max_words]
    return key_words


def find_word_position_in_context(context: str, word: str) -> int:
    """
    Находит позицию слова в контексте (в символах).
    Используется для пересчёта marker_pos после изменения контекста.

    Ищет слово как отдельное (с границами слов), не как подстроку.
    Возвращает -1 если слово не найдено.
    """
    if not context or not word:
        return -1

    import re
    context_norm = normalize_for_search(context)
    word_norm = normalize_for_search(word)

    # Экранируем спецсимволы regex
    escaped = re.escape(word_norm)

    # Ищем слово с границами (начало строки или пробел/пунктуация)
    pattern = r'(^|[\s,.:;!?\-—«»""()\n])(' + escaped + r')([\s,.:;!?\-—«»""()\n]|$)'
    match = re.search(pattern, context_norm)

    if match:
        # Позиция — начало самого слова (после разделителя)
        return match.start() + len(match.group(1))

    # Fallback: простой поиск подстроки
    idx = context_norm.find(word_norm)
    return idx


def setup_logging():
    """Настройка логирования в файл и консоль с ротацией"""
    global LOG_DIR

    # Используем путь из config.py если доступен
    if HAS_CONFIG:
        LOG_DIR = TEMP_DIR / 'web_logs'
    else:
        project_dir = Path(__file__).parent.parent
        LOG_DIR = project_dir / 'Темп' / 'web_logs'

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Ротация логов: удаляем старые (оставляем только 10 последних)
    cleanup_old_logs(LOG_DIR, max_files=10)

    # Файл лога с датой
    log_file = LOG_DIR / f'web_viewer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # Настраиваем формат
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Файловый handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Консольный handler (только важные сообщения)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Создаём логгер
    logger = logging.getLogger('web_viewer')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f'=== Web Viewer v3.0 запущен ===')
    logger.info(f'Лог-файл: {log_file}')

    return logger


def cleanup_old_logs(log_dir: Path, max_files: int = 10):
    """Удаляет старые лог-файлы, оставляя только max_files последних"""
    log_files = sorted(log_dir.glob('web_viewer_*.log'), reverse=True)

    # Удаляем файлы сверх лимита
    for old_file in log_files[max_files:]:
        try:
            old_file.unlink()
        except Exception:
            pass  # Игнорируем ошибки удаления

# Инициализируем логгер
log = setup_logging()

def log_enrichment(error_idx, error_type, operation, details):
    """Логирует операцию обогащения контекста"""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'error_idx': error_idx,
        'error_type': error_type,
        'operation': operation,
        'details': details
    }
    DEBUG_STATE['enrichment_log'].append(entry)
    log.debug(f'[E{error_idx}] {error_type} | {operation}: {details}')

def save_debug_state():
    """Сохраняет текущее состояние отладки в файл"""
    global LOG_DIR
    if not LOG_DIR:
        return

    debug_file = LOG_DIR / 'debug_state.json'
    try:
        # Подготавливаем данные для сериализации
        state_to_save = {
            'timestamp': datetime.now().isoformat(),
            'original_loaded': DEBUG_STATE['original_loaded'],
            'transcript_loaded': DEBUG_STATE['transcript_loaded'],
            'errors_count': len(DEBUG_STATE['errors']),
            'enrichment_log': DEBUG_STATE['enrichment_log'][-50:],  # Последние 50 записей
            'last_request': DEBUG_STATE['last_request'],
        }
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)
        log.debug(f'Debug state saved to {debug_file}')
    except Exception as e:
        log.error(f'Failed to save debug state: {e}')

# Порт по умолчанию
DEFAULT_PORT = 8765

# Путь к HTML шаблону
TEMPLATE_DIR = Path(__file__).parent / 'templates'
TEMPLATE_FILE = TEMPLATE_DIR / 'viewer.html'


def load_html_template():
    """Загружает HTML шаблон из файла"""
    if TEMPLATE_FILE.exists():
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        log.error(f'Файл шаблона не найден: {TEMPLATE_FILE}')
        raise FileNotFoundError(f'HTML шаблон не найден: {TEMPLATE_FILE}. Убедитесь, что templates/viewer.html существует.')


# Загружаем шаблон из файла при импорте модуля
HTML_TEMPLATE = load_html_template()


# Глобальный путь для сохранения ложных ошибок
FALSE_POSITIVES_FILE = None
# Глобальный путь для сохранения ошибок чтецу
READER_ERRORS_FILE = None
# Папка для файлов чтецу
READER_OUTPUT_DIR = None
# Номер главы из файла
CHAPTER_NUMBER = None


class CustomHandler(SimpleHTTPRequestHandler):
    """Обработчик HTTP запросов с поддержкой CORS, аудио и API"""

    def __init__(self, *args, directory=None, **kwargs):
        self.base_directory = directory or os.getcwd()
        super().__init__(*args, directory=self.base_directory, **kwargs)

    def end_headers(self):
        # Добавляем CORS заголовки для локальной разработки
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_POST(self):
        """Обработка POST запросов для сохранения ошибок"""
        if self.path == '/api/false-positive':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                error_data = json.loads(post_data.decode('utf-8'))
                save_false_positive(error_data)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

        elif self.path == '/api/reader-error':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                error_data = json.loads(post_data.decode('utf-8'))
                save_reader_error(error_data)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

        elif self.path == '/api/download-reader-docx':
            print(f"  → POST /api/download-reader-docx получен")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                errors = data.get('errors', [])
                print(f"  → Генерация DOCX для {len(errors)} ошибок...")
                print(f"  → READER_OUTPUT_DIR = {READER_OUTPUT_DIR}")
                print(f"  → CHAPTER_NUMBER = {CHAPTER_NUMBER}")
                filename = generate_reader_docx(errors)
                print(f"  → Файл создан: {filename}")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'file': filename}).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        # API для отладки
        if self.path == '/api/debug':
            DEBUG_STATE['last_request'] = datetime.now().isoformat()
            save_debug_state()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            # Формируем отладочную информацию
            debug_info = {
                'timestamp': datetime.now().isoformat(),
                'original_loaded': DEBUG_STATE['original_loaded'],
                'transcript_loaded': DEBUG_STATE['transcript_loaded'],
                'errors_count': len(DEBUG_STATE['errors']),
                'enrichment_summary': {},
                'recent_log': DEBUG_STATE['enrichment_log'][-20:],
                'log_dir': str(LOG_DIR) if LOG_DIR else None,
            }

            # Суммируем операции обогащения
            for entry in DEBUG_STATE['enrichment_log']:
                op = entry.get('operation', 'unknown')
                debug_info['enrichment_summary'][op] = debug_info['enrichment_summary'].get(op, 0) + 1

            self.wfile.write(json.dumps(debug_info, ensure_ascii=False, indent=2).encode())
            return

        # API для получения лог-файла
        if self.path == '/api/log':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()

            if LOG_DIR:
                # Находим последний лог-файл
                log_files = sorted(LOG_DIR.glob('web_viewer_*.log'), reverse=True)
                if log_files:
                    with open(log_files[0], 'r', encoding='utf-8') as f:
                        # Последние 200 строк
                        lines = f.readlines()[-200:]
                        self.wfile.write(''.join(lines).encode())
                        return

            self.wfile.write(b'No log file found')
            return

        # Обработка скачивания DOCX файла
        if self.path.startswith('/download/'):
            filename = urllib.parse.unquote(self.path[10:])  # убираем /download/
            print(f"  → GET /download/{filename}")
            print(f"  → READER_OUTPUT_DIR = {READER_OUTPUT_DIR}")

            filepath = os.path.join(READER_OUTPUT_DIR, filename) if READER_OUTPUT_DIR else None
            print(f"  → filepath = {filepath}")
            print(f"  → exists = {os.path.exists(filepath) if filepath else False}")

            if filepath and os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                # Кодируем имя файла для заголовка
                encoded_filename = urllib.parse.quote(filename)
                self.send_header('Content-Disposition', f"attachment; filename*=UTF-8''{encoded_filename}")
                self.send_header('Content-Length', str(os.path.getsize(filepath)))
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
                print(f"  → Файл отправлен: {filepath}")
                return
            else:
                print(f"  → ОШИБКА: Файл не найден!")
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f'Файл не найден: {filename}'.encode('utf-8'))
                return

        # Обработка запросов к аудиофайлам с полной поддержкой Range (HTTP 206)
        audio_extensions = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg'}
        for ext, content_type in audio_extensions.items():
            if self.path.endswith(ext):
                # Получаем путь к файлу
                filename = urllib.parse.unquote(self.path.lstrip('/'))
                filepath = os.path.join(self.directory, filename)

                if not os.path.exists(filepath):
                    self.send_response(404)
                    self.end_headers()
                    return

                file_size = os.path.getsize(filepath)

                # Проверяем Range заголовок
                range_header = self.headers.get('Range')

                if range_header:
                    # Парсим Range: bytes=start-end
                    range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
                    if range_match:
                        start = int(range_match.group(1)) if range_match.group(1) else 0
                        end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

                        # Ограничиваем end
                        end = min(end, file_size - 1)
                        content_length = end - start + 1

                        # Отправляем 206 Partial Content
                        self.send_response(206)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Accept-Ranges', 'bytes')
                        self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                        self.send_header('Content-Length', str(content_length))
                        self.end_headers()

                        # Читаем и отправляем нужную часть файла
                        with open(filepath, 'rb') as f:
                            f.seek(start)
                            self.wfile.write(f.read(content_length))
                        return

                # Без Range — отправляем весь файл
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(file_size))
                self.end_headers()

                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
                return

        super().do_GET()


def save_false_positive(error_data):
    """Сохраняет ложную ошибку в файл"""
    global FALSE_POSITIVES_FILE

    if not FALSE_POSITIVES_FILE:
        FALSE_POSITIVES_FILE = os.path.join(os.getcwd(), 'ложные_ошибки.json')

    # Загружаем существующие ошибки
    existing = []
    if os.path.exists(FALSE_POSITIVES_FILE):
        try:
            with open(FALSE_POSITIVES_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            existing = []

    # Добавляем новую
    existing.append(error_data)

    # Сохраняем
    with open(FALSE_POSITIVES_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"  → Ложная ошибка сохранена: {error_data.get('time_formatted', '?')} — {error_data.get('type', '?')}")


def save_reader_error(error_data):
    """Сохраняет ошибку для чтеца в файл"""
    global READER_ERRORS_FILE

    if not READER_ERRORS_FILE:
        READER_ERRORS_FILE = os.path.join(os.getcwd(), 'ошибки_чтецу.json')

    # Загружаем существующие ошибки
    existing = []
    if os.path.exists(READER_ERRORS_FILE):
        try:
            with open(READER_ERRORS_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            existing = []

    # Добавляем новую
    existing.append(error_data)

    # Сохраняем
    with open(READER_ERRORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"  → Ошибка для чтеца: {error_data.get('time_formatted', '?')} — {error_data.get('type', '?')}")


def generate_reader_docx(errors):
    """
    Генерирует DOCX файл для чтеца.
    Делегирует в модуль docx_export.py (выделен в v5.1.0).
    """
    global READER_OUTPUT_DIR, CHAPTER_NUMBER

    # Определяем папку
    if not READER_OUTPUT_DIR:
        if HAS_CONFIG:
            READER_OUTPUT_DIR = str(READER_DIR)
        else:
            project_dir = Path(__file__).parent.parent
            READER_OUTPUT_DIR = str(project_dir / 'Чтецу')

    chapter_num = CHAPTER_NUMBER or '1'

    # Определяем имя файла
    if HAS_CONFIG:
        filename = FileNaming.build_filename(chapter_num, 'docx')
    else:
        filename = f'Чтецу_глава_{chapter_num}.docx'

    if HAS_DOCX_EXPORT:
        return _generate_reader_docx_module(
            errors=errors,
            output_dir=READER_OUTPUT_DIR,
            chapter_number=chapter_num,
            filename=filename
        )
    else:
        # Fallback: встроенная упрощённая генерация TXT
        os.makedirs(READER_OUTPUT_DIR, exist_ok=True)
        txt_filename = f'Чтецу_глава_{chapter_num}.txt'
        txt_filepath = os.path.join(READER_OUTPUT_DIR, txt_filename)
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            for error in errors:
                time_str = error.get('time_formatted', '0:00')
                error_type = error.get('type', '')
                f.write(f'{time_str} [{error_type}]\n')
        print(f"  -> TXT для чтеца создан: {txt_filepath}")
        return txt_filename


def load_original_text(file_path):
    """Загружает оригинальный текст из DOCX или TXT с пунктуацией"""
    log.info(f'load_original_text: {file_path}')

    try:
        # Определяем формат по расширению
        if file_path.lower().endswith('.txt'):
            log.debug(f'  → Формат: TXT')
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                log.info(f'  ✓ Загружено {len(text)} символов из TXT')
                DEBUG_STATE['original_loaded'] = True
                return text
        else:
            # DOCX формат
            log.debug(f'  → Формат: DOCX')
            from docx import Document
            doc = Document(file_path)
            text = '\n'.join(p.text for p in doc.paragraphs)
            log.info(f'  ✓ Загружено {len(text)} символов из DOCX')
            DEBUG_STATE['original_loaded'] = True
            return text
    except ImportError:
        log.error('  ✗ python-docx не установлен')
        print("  ⚠ python-docx не установлен")
        return None
    except Exception as e:
        log.error(f'  ✗ Ошибка загрузки: {e}')
        print(f"  ⚠ Ошибка загрузки {file_path}: {e}")
        return None


def find_original_context(original_text, error, context_chars=120):
    """
    Находит контекст в оригинальном тексте по ключевым словам из ошибки.
    Возвращает короткий фрагмент (~120 символов) с пунктуацией.

    ВАЖНО: Использует единые правила SHORT_WORD_THRESHOLD и PRIORITY_WORD_MIN_LENGTH
    для определения, когда перезаписывать контекст, а когда оставить оригинальный.
    """
    if not original_text:
        return error.get('context', '')

    norm_context = error.get('context', '')
    if not norm_context:
        return ''

    text_normalized = normalize_for_search(original_text)

    # Получаем ключевое слово из ошибки
    error_word = ''
    error_phrase = ''  # Для transposition — полная фраза

    if error.get('type') == 'substitution':
        error_word = error.get('correct', '')
    elif error.get('type') == 'deletion':
        error_word = error.get('word', '') or error.get('correct', '')
    elif error.get('type') == 'insertion':
        # Для insertion ищем по соседним словам из контекста
        error_word = ''
    elif error.get('type') == 'transposition':
        # Для перестановки берём правильную фразу целиком (например "что ты")
        error_phrase = error.get('correct', '') or error.get('original', '')
        # Отдельные слова могут быть короткими, но фраза — уникальная
        if error_phrase:
            error_word = error_phrase  # Используем всю фразу как ключ

    # ЕДИНОЕ ПРАВИЛО: Если слово ошибки короткое, не перезаписываем контекст
    # Используем константу SHORT_WORD_THRESHOLD (по умолчанию 4)
    # Исключение: transposition с длинной фразой (error_phrase)
    if error_word and is_short_word(error_word) and not error_phrase:
        return norm_context  # Оставляем оригинальный контекст из JSON

    # Берём ключевые слова из контекста (используем единую функцию)
    key_words = get_key_words_from_context(norm_context)

    # Если есть слово ошибки достаточной длины, добавляем его в начало
    # Используем константу PRIORITY_WORD_MIN_LENGTH (по умолчанию 5)
    if error_word and len(error_word) >= PRIORITY_WORD_MIN_LENGTH:
        key_words = [error_word] + key_words

    # Ищем позицию с максимальным совпадением ключевых слов
    best_pos = -1
    best_score = 0

    for i in range(0, len(text_normalized) - 100, 10):
        chunk = text_normalized[i:i+200]
        score = 0
        for j, w in enumerate(key_words):
            if normalize_for_search(w) in chunk:
                # Первые слова (особенно слово ошибки) важнее
                score += (len(key_words) - j)
        if score > best_score:
            best_score = score
            best_pos = i

    if best_pos >= 0 and best_score >= 3:
        # Находим точную позицию слова ошибки (приоритет) или ключевого слова
        chunk = text_normalized[best_pos:best_pos+200]
        key_offset = 0

        # Сначала ищем именно слово ошибки (correct для substitution)
        if error_word:
            idx = chunk.find(normalize_for_search(error_word))
            if idx != -1:
                key_offset = idx

        # Если не нашли error_word, ищем по key_words
        if key_offset == 0:
            for w in key_words:
                idx = chunk.find(normalize_for_search(w))
                if idx != -1:
                    key_offset = idx
                    break

        # Центрируем контекст вокруг найденного слова
        center = best_pos + key_offset
        half = context_chars // 2
        start = max(0, center - half)
        end = min(len(original_text), center + half)

        # Подгоняем к границам слов
        while start > 0 and original_text[start] not in ' \n':
            start -= 1
        while end < len(original_text) and original_text[end] not in ' \n':
            end += 1

        context = original_text[start:end].strip()
        context = ' '.join(context.split())
        return context

    return error.get('context', '')


def load_transcript_words(transcript_path):
    """Загружает слова из транскрипции с временными метками"""
    log.info(f'load_transcript_words: {transcript_path}')

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = []
        for chunk in data.get('chunks', []):
            for alt in chunk.get('alternatives', []):
                for w in alt.get('words', []):
                    word = w.get('word', '')
                    start = float(w.get('startTime', '0').replace('s', ''))
                    words.append((start, word))

        words.sort(key=lambda x: x[0])
        log.info(f'  ✓ Загружено {len(words)} слов из транскрипции')
        DEBUG_STATE['transcript_loaded'] = True
        return words
    except Exception as e:
        log.error(f'  ✗ Ошибка загрузки транскрипции: {e}')
        print(f"  ⚠ Ошибка загрузки транскрипции: {e}")
        return []


def find_insertion_neighbors(transcript_words, error_time, error_word):
    """Находит соседние слова для insertion в транскрипции.

    Ищет наиболее близкое по времени совпадение, а не первое найденное.
    Это важно когда одинаковые слова встречаются рядом (например, два "не").
    """
    log.debug(f'  find_insertion_neighbors: time={error_time:.1f}, word="{error_word}"')

    # Собираем все совпадения по слову в пределах ±1.5 сек
    candidates = []
    for i, (t, w) in enumerate(transcript_words):
        if abs(t - error_time) < 1.5 and w.lower() == error_word.lower():
            candidates.append((abs(t - error_time), i, t))

    if candidates:
        # Выбираем наиболее близкое по времени
        candidates.sort(key=lambda x: x[0])
        _, best_idx, best_time = candidates[0]

        before = transcript_words[best_idx-1][1] if best_idx > 0 else ''
        after = transcript_words[best_idx+1][1] if best_idx < len(transcript_words)-1 else ''
        log.debug(f'    → Найдено (из {len(candidates)} кандидатов): idx={best_idx}, t={best_time:.1f}, before="{before}", after="{after}"')
        return before, after

    # Если не нашли точное совпадение, показываем ближайшие слова для отладки
    nearby = [(t, w) for t, w in transcript_words if abs(t - error_time) < 3]
    if nearby:
        log.debug(f'    → Не найдено "{error_word}". Ближайшие слова: {nearby[:5]}')

    return '', ''


def find_deletion_neighbors(transcript_words, error_time):
    """Находит соседние слова для deletion в транскрипции (слова до и после пропуска)

    Для deletion время ошибки — это момент, когда должно было звучать пропущенное слово.
    Нужно найти:
    - before_word: последнее слово ДО момента ошибки
    - after_word: первое слово В момент ошибки или сразу ПОСЛЕ
    """
    before_word = ''
    after_word = ''

    # Небольшой буфер (0.1 сек) для определения "строго до"
    epsilon = 0.1

    for i, (t, w) in enumerate(transcript_words):
        if t < error_time - epsilon:
            # Слово строго ДО момента ошибки
            before_word = w
        elif t >= error_time - epsilon and not after_word:
            # Первое слово В момент ошибки или ПОСЛЕ — это то, что произнесено вместо пропущенного
            after_word = w
            break

    return before_word, after_word


def find_insertion_context_in_original(original_text, before_word, after_word, context_chars=180):
    """
    Находит контекст в оригинале по соседним словам и ставит маркер между ними.
    Для insertion ошибок берём больше текста ПОСЛЕ маркера, чтобы было видно контекст.
    Использует нечёткий поиск для учёта разных форм слов (пинка/пинками).
    """
    log.debug(f'  find_insertion_context: before="{before_word}", after="{after_word}"')

    if not original_text or not before_word or not after_word:
        log.debug(f'    → Пропуск: missing data')
        return None, -1

    import re
    text_lower = original_text.lower().replace('ё', 'е')
    before_lower = before_word.lower().replace('ё', 'е')
    after_lower = after_word.lower().replace('ё', 'е')

    # Ищем паттерн "before_word ... after_word" (точное совпадение)
    # Разрешаем между словами пунктуацию, переводы строк, тире, кавычки
    sep = r'[\s,.:;!?\-—–«»""()\n\r…]*'
    pattern = before_lower + sep + after_lower
    log.debug(f'    → Pattern: {pattern}')

    match = re.search(pattern, text_lower)

    # Если не нашли точно, пробуем нечёткий поиск (основа слова + любое окончание)
    if not match:
        # Обрезаем окончания (простая эвристика для русских слов)
        # Берём основу слова (минимум 3 символа) и ищем с любым окончанием
        before_stem = before_lower[:max(3, len(before_lower)-3)] if len(before_lower) > 3 else before_lower
        after_stem = after_lower[:max(3, len(after_lower)-3)] if len(after_lower) > 3 else after_lower

        # Паттерн: основа + любые буквы + пунктуация/переводы строк + основа + любые буквы
        fuzzy_sep = r'[а-яё]*[\s,.:;!?\-—–«»""()\n\r…]+'
        fuzzy_pattern = before_stem + fuzzy_sep + after_stem + r'[а-яё]*'
        log.debug(f'    → Fuzzy pattern: {fuzzy_pattern}')
        match = re.search(fuzzy_pattern, text_lower)

    if match:
        # Нашли! Позиция маркера — после первого слова
        # Ищем конец первого слова в совпадении
        matched_text = match.group()
        first_word_end = 0
        for i, c in enumerate(matched_text):
            if c in ' ,.\t\n-—':
                first_word_end = i
                break

        marker_pos = match.start() + first_word_end
        log.debug(f'    → Match found at {match.start()}, marker_pos={marker_pos}')

        # Вырезаем контекст асимметрично: 1/3 до маркера, 2/3 после
        # Это даёт больше текста после ошибки
        before_chars = context_chars // 3
        after_chars = context_chars - before_chars

        start = max(0, match.start() - before_chars)
        end = min(len(original_text), match.end() + after_chars)

        # Подгоняем к границам слов
        while start > 0 and original_text[start] not in ' \n':
            start -= 1
        while end < len(original_text) and original_text[end] not in ' \n':
            end += 1

        context = original_text[start:end].strip()
        context = ' '.join(context.split())

        # Позиция маркера относительно начала контекста
        marker_rel_pos = marker_pos - start
        log.debug(f'    → Context extracted: start={start}, marker_rel_pos={marker_rel_pos}')
        log.debug(f'    → Context: "{context[:60]}..."')

        return context, marker_rel_pos

    log.debug(f'    → Pattern not found in text')
    return None, -1


def find_deletion_context_in_original(original_text, before_word, after_word, missing_word, context_chars=120):
    """
    Находит контекст в оригинале для deletion.
    Ищет паттерн "before_word + missing_word + after_word" и возвращает контекст,
    содержащий пропущенное слово.
    """
    if not original_text or not missing_word:
        return None

    import re
    text_lower = original_text.lower().replace('ё', 'е')
    missing_lower = missing_word.lower().replace('ё', 'е')

    # Если есть соседи — ищем более точно
    if before_word and after_word:
        before_lower = before_word.lower().replace('ё', 'е')
        after_lower = after_word.lower().replace('ё', 'е')

        # Паттерн: before + пропущенное + after
        # Разделитель включает пунктуацию, переводы строк, тире, кавычки — всё что может быть между словами в оригинале
        sep = r'[\s,.:;!?\-—–«»""()\n\r…]*'
        pattern = before_lower + sep + missing_lower + sep + after_lower
        match = re.search(pattern, text_lower)

        if match:
            half = context_chars // 2
            start = max(0, match.start() - half)
            end = min(len(original_text), match.end() + half)

            while start > 0 and original_text[start] not in ' \n':
                start -= 1
            while end < len(original_text) and original_text[end] not in ' \n':
                end += 1

            context = original_text[start:end].strip()
            context = ' '.join(context.split())
            return context

    # Fallback: ищем просто пропущенное слово
    pattern = r'(^|[\s,.:;!?\-—«»""()\n])' + re.escape(missing_lower) + r'([\s,.:;!?\-—«»""()\n]|$)'
    match = re.search(pattern, text_lower)

    if match:
        center = match.start()
        half = context_chars // 2
        start = max(0, center - half)
        end = min(len(original_text), center + half + len(missing_word))

        while start > 0 and original_text[start] not in ' \n':
            start -= 1
        while end < len(original_text) and original_text[end] not in ' \n':
            end += 1

        context = original_text[start:end].strip()
        context = ' '.join(context.split())
        return context

    return None


def enrich_errors_with_original(errors, original_text, transcript_words=None):
    """Заменяет контексты на оригинальные (с пунктуацией)"""
    log.info(f'enrich_errors_with_original: {len(errors)} ошибок')
    log.debug(f'  original_text: {len(original_text) if original_text else 0} символов')
    log.debug(f'  transcript_words: {len(transcript_words) if transcript_words else 0} слов')

    if not original_text:
        log.warning('  ✗ Нет оригинального текста — обогащение пропущено')
        return errors

    # Сохраняем в debug state
    DEBUG_STATE['errors'] = errors

    for idx, error in enumerate(errors):
        error_time = error.get('time', 0)
        time_str = error.get('time', '?')
        if isinstance(error_time, str):
            parts = error_time.split(':')
            error_time = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else 0

        error_type = error.get('type', 'unknown')

        if error_type == 'insertion':
            # Для insertion: ищем соседей в транскрипции, затем место в оригинале
            error_word = error.get('word', '') or error.get('transcript', '')
            existing_context = error.get('context', '')
            log_enrichment(idx, 'insertion', 'start', f'word="{error_word}", time={time_str}')

            # ЕДИНОЕ ПРАВИЛО: если контекст уже есть, достаточно длинный И имеет пунктуацию — не перезаписываем
            # Для insertion проверяем, что контекст не пустой и достаточно длинный
            has_punctuation = any(c in existing_context for c in '.,!?;:—–-«»""()') if existing_context else False
            if existing_context and len(existing_context) > 50 and has_punctuation:
                log_enrichment(idx, 'insertion', 'skipped', f'context already exists with punctuation (len={len(existing_context)})')
                continue

            if transcript_words:
                before, after = find_insertion_neighbors(transcript_words, error_time, error_word)
                log_enrichment(idx, 'insertion', 'find_neighbors', f'before="{before}", after="{after}"')

                if before and after:
                    context, marker_pos = find_insertion_context_in_original(original_text, before, after)
                    log_enrichment(idx, 'insertion', 'find_context', f'marker_pos={marker_pos}, context_len={len(context) if context else 0}')

                    if context and marker_pos >= 0:
                        error['context'] = context
                        error['marker_pos'] = marker_pos
                        error['neighbors'] = f"{before}|{after}"
                        log_enrichment(idx, 'insertion', 'success', f'context="{context[:50]}..."')
                    else:
                        log_enrichment(idx, 'insertion', 'failed', 'context not found in original')
                else:
                    log_enrichment(idx, 'insertion', 'failed', 'neighbors not found in transcript')
            else:
                log_enrichment(idx, 'insertion', 'skipped', 'no transcript_words')
            continue

        if error_type == 'deletion':
            # Для deletion: ищем контекст в оригинале, содержащий пропущенное слово
            missing_word = error.get('word', '') or error.get('correct', '')
            log_enrichment(idx, 'deletion', 'start', f'word="{missing_word}", time={time_str}')

            # ЕДИНОЕ ПРАВИЛО: для коротких слов не перезаписываем контекст
            if is_short_word(missing_word):
                log_enrichment(idx, 'deletion', 'skipped', f'short word "{missing_word}" (len={len(missing_word)} <= {SHORT_WORD_THRESHOLD})')
                continue

            before, after = '', ''
            if transcript_words:
                before, after = find_deletion_neighbors(transcript_words, error_time)
                log_enrichment(idx, 'deletion', 'find_neighbors', f'before="{before}", after="{after}"')

            # Ищем контекст с пропущенным словом
            context = find_deletion_context_in_original(original_text, before, after, missing_word)
            if context:
                error['context'] = context
                if before and after:
                    error['neighbors'] = f"{before}|{after}"
                # Пересчитываем marker_pos для нового контекста
                new_marker_pos = find_word_position_in_context(context, missing_word)
                if new_marker_pos >= 0:
                    error['marker_pos'] = new_marker_pos
                log_enrichment(idx, 'deletion', 'success', f'context="{context[:50]}...", marker_pos={new_marker_pos}')
            else:
                log_enrichment(idx, 'deletion', 'failed', 'context not found')
            continue

        # substitution
        if error_type == 'substitution':
            # Поддержка обоих форматов: correct/wrong и original/transcript
            correct_word = error.get('correct', '') or error.get('original', '')
            wrong_word = error.get('wrong', '') or error.get('transcript', '')
            existing_context = error.get('context', '')
            log_enrichment(idx, 'substitution', 'start', f'wrong="{wrong_word}", correct="{correct_word}", time={time_str}')

            # ЕДИНОЕ ПРАВИЛО: если контекст уже содержит слово ошибки И имеет пунктуацию — не перезаписываем
            # Это предотвращает замену правильного контекста на неправильный при повторяющихся словах
            # Но если контекст без пунктуации (из нормализованного файла) — заменяем на оригинальный
            has_punctuation = any(c in existing_context for c in '.,!?;:—–-«»""()') if existing_context else False
            if existing_context and correct_word and has_punctuation:
                if normalize_for_search(correct_word) in normalize_for_search(existing_context):
                    log_enrichment(idx, 'substitution', 'skipped', f'context already contains "{correct_word}" with punctuation')
                    continue

            original_context = find_original_context(original_text, error)
            if original_context and len(original_context) > 20:
                error['context'] = original_context
                # Пересчитываем marker_pos для нового контекста
                new_marker_pos = find_word_position_in_context(original_context, correct_word)
                if new_marker_pos >= 0:
                    error['marker_pos'] = new_marker_pos
                log_enrichment(idx, 'substitution', 'success', f'context="{original_context[:50]}...", marker_pos={new_marker_pos}')
            else:
                log_enrichment(idx, 'substitution', 'failed', 'context not found or too short')

        # transposition (перестановка слов)
        if error_type == 'transposition':
            # Поддержка обоих форматов: correct/wrong и original/transcript
            correct_phrase = error.get('correct', '') or error.get('original', '')
            wrong_phrase = error.get('wrong', '') or error.get('transcript', '')
            existing_context = error.get('context', '')
            log_enrichment(idx, 'transposition', 'start', f'wrong="{wrong_phrase}", correct="{correct_phrase}", time={time_str}')

            # ЕДИНОЕ ПРАВИЛО: если контекст уже содержит фразу ошибки И имеет пунктуацию — не перезаписываем
            has_punctuation = any(c in existing_context for c in '.,!?;:—–-«»""()') if existing_context else False
            if existing_context and correct_phrase and has_punctuation:
                if normalize_for_search(correct_phrase) in normalize_for_search(existing_context):
                    log_enrichment(idx, 'transposition', 'skipped', f'context already contains "{correct_phrase}" with punctuation')
                    continue

            original_context = find_original_context(original_text, error)
            if original_context and len(original_context) > 20:
                error['context'] = original_context
                # Пересчитываем marker_pos для нового контекста
                new_marker_pos = find_word_position_in_context(original_context, correct_phrase)
                if new_marker_pos >= 0:
                    error['marker_pos'] = new_marker_pos
                log_enrichment(idx, 'transposition', 'success', f'context="{original_context[:50]}...", marker_pos={new_marker_pos}')
            else:
                log_enrichment(idx, 'transposition', 'failed', 'context not found or too short')

    # Сохраняем debug state после обогащения
    save_debug_state()
    log.info(f'  ✓ Обогащение завершено')

    return errors


def load_json_results(json_path):
    """Загрузка результатов из JSON файла

    Поддерживает два формата полей:
    - Старый: wrong/correct
    - Новый: transcript/original
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Преобразуем формат ошибок если нужно
    errors = data.get('errors', [])
    converted = []

    for e in errors:
        err = {
            'time': e.get('time', e.get('time_formatted', '0:00')),
            'type': e.get('type', 'substitution'),
            'context': e.get('context', ''),
            'marker_pos': e.get('marker_pos', -1),
        }

        # Поддержка обоих форматов: wrong/correct и transcript/original
        # transcript = что услышал Яндекс (ошибка) = wrong
        # original = что должно быть = correct
        wrong_val = e.get('wrong') or e.get('transcript', '')
        correct_val = e.get('correct') or e.get('original', '')

        if err['type'] == 'substitution':
            err['wrong'] = wrong_val
            err['correct'] = correct_val
        elif err['type'] == 'transposition':
            # transposition: перестановка слов
            # wrong = что услышал Яндекс (неправильный порядок)
            # correct = что должно быть (правильный порядок)
            err['wrong'] = wrong_val
            err['correct'] = correct_val
        elif err['type'] == 'insertion':
            # Лишнее слово — в поле wrong/transcript
            err['word'] = wrong_val
        elif err['type'] == 'deletion':
            # Пропущенное слово — в поле correct/original
            err['word'] = correct_val

        converted.append(err)

    # Сортируем ошибки по времени
    def get_time_seconds(err):
        t = err.get('time', 0)
        if isinstance(t, (int, float)):
            return t
        if isinstance(t, str) and ':' in t:
            parts = t.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        return 0

    converted.sort(key=get_time_seconds)

    return {'errors': converted, 'audio': data.get('audio')}


def convert_docx_errors_to_json(errors_from_check):
    """Конвертация ошибок из check_transcription.py в формат для веб-интерфейса"""
    json_errors = []

    for error in errors_from_check:
        json_error = {
            'time': error.get('timecode', '0:00'),
            'type': error.get('type', 'substitution'),
            'context': error.get('context', '')
        }

        if error['type'] == 'substitution':
            json_error['wrong'] = error.get('word1', '')
            json_error['correct'] = error.get('word2', '')
        elif error['type'] == 'insertion':
            json_error['word'] = error.get('word', '')
        elif error['type'] == 'deletion':
            json_error['word'] = error.get('word', '')

        json_errors.append(json_error)

    return json_errors


def escape_error_data(errors):
    """Экранирование HTML в данных ошибок для защиты от XSS"""
    safe_errors = []
    for error in errors:
        safe_error = {}
        for key, value in error.items():
            if isinstance(value, str):
                # Экранируем HTML-специальные символы
                safe_error[key] = html.escape(value)
            else:
                safe_error[key] = value
        safe_errors.append(safe_error)
    return safe_errors


def generate_html(errors, audio_path=None):
    """Генерация HTML страницы с данными"""
    # Экранируем данные для защиты от XSS
    safe_errors = escape_error_data(errors)
    safe_audio = html.escape(audio_path) if audio_path else None

    data = {
        'errors': safe_errors,
        'audio': safe_audio
    }

    html_content = HTML_TEMPLATE.replace('__ERROR_DATA__', json.dumps(data, ensure_ascii=False))
    return html_content


def run_server(html_content, audio_path=None, port=DEFAULT_PORT, json_file=None):
    """Запуск веб-сервера"""
    global FALSE_POSITIVES_FILE, READER_ERRORS_FILE, READER_OUTPUT_DIR, CHAPTER_NUMBER

    # Создаем временный HTML файл
    temp_dir = Path('/tmp/transcription_viewer')
    temp_dir.mkdir(exist_ok=True)

    html_file = temp_dir / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Если есть аудио, создаем симлинк
    if audio_path:
        audio_link = temp_dir / Path(audio_path).name
        # Удаляем старый симлинк/файл (exists не видит битые симлинки, поэтому проверяем и is_symlink)
        if audio_link.exists() or audio_link.is_symlink():
            audio_link.unlink()
        try:
            os.symlink(os.path.abspath(audio_path), audio_link)
        except OSError:
            # Если симлинк не работает, копируем файл
            import shutil
            shutil.copy(audio_path, audio_link)

    # Запуск сервера
    os.chdir(temp_dir)

    # Определяем номер главы из имени файла через FileNaming
    if json_file:
        if HAS_CONFIG:
            # Используем унифицированный метод из config.py
            CHAPTER_NUMBER = FileNaming.get_chapter_id(Path(json_file))
        else:
            # Fallback: старая логика
            import re
            match = re.search(r'глава(\d+)', json_file, re.IGNORECASE)
            if match:
                CHAPTER_NUMBER = match.group(1)
            else:
                match = re.search(r'(\d+)', os.path.basename(json_file))
                CHAPTER_NUMBER = match.group(1) if match else '1'
    else:
        CHAPTER_NUMBER = '1'

    # Устанавливаем пути для сохранения файлов
    if HAS_CONFIG:
        # Используем централизованные пути из config.py
        ensure_dirs_exist()
        FALSE_POSITIVES_FILE = str(RESULTS_DIR / 'ложные_ошибки.json')
        READER_ERRORS_FILE = str(RESULTS_DIR / 'ошибки_чтецу.json')
        READER_OUTPUT_DIR = str(READER_DIR)
    else:
        # Fallback: вычисляем пути вручную
        project_dir = Path(__file__).parent.parent
        FALSE_POSITIVES_FILE = str(project_dir / 'Результаты проверки' / 'ложные_ошибки.json')
        READER_ERRORS_FILE = str(project_dir / 'Результаты проверки' / 'ошибки_чтецу.json')
        READER_OUTPUT_DIR = str(project_dir / 'Чтецу')

    # Очищаем файл ложных ошибок при каждом запуске новой сессии
    with open(FALSE_POSITIVES_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)
    print(f"  ✓ Файл ложных ошибок очищен (новая сессия)")

    # Создаём папку Чтецу если её нет
    os.makedirs(READER_OUTPUT_DIR, exist_ok=True)

    handler = lambda *args, **kwargs: CustomHandler(*args, directory=str(temp_dir), **kwargs)
    server = HTTPServer(('localhost', port), handler)

    url = f'http://localhost:{port}'
    print(f"\n{'='*60}")
    print(f"  Веб-интерфейс запущен: {url}")
    print(f"  Глава: {CHAPTER_NUMBER}")
    print(f"  Ложные ошибки → {FALSE_POSITIVES_FILE}")
    print(f"  Файлы для чтеца → {READER_OUTPUT_DIR}")
    print(f"")
    print(f"  === Debug API ===")
    print(f"  {url}/api/debug  — состояние и логи обогащения (JSON)")
    print(f"  {url}/api/log    — последние 200 строк лог-файла")
    print(f"  Логи: {LOG_DIR}")
    print(f"")
    print(f"  Нажмите Ctrl+C для остановки")
    print(f"{'='*60}\n")

    # Открываем браузер
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен")
        server.shutdown()


def demo_errors():
    """Демо-данные для тестирования"""
    return [
        {'time': '0:29', 'type': 'substitution', 'wrong': 'ОРДЕНА', 'correct': 'Орден', 'context': 'магистр Ордена'},
        {'time': '1:38', 'type': 'substitution', 'wrong': 'ВОЗОМНИЛА', 'correct': 'возомнил', 'context': 'он возомнил себя'},
        {'time': '2:52', 'type': 'substitution', 'wrong': 'ШАУГАД', 'correct': 'Шаугат', 'context': 'господин Шаугат'},
        {'time': '3:58', 'type': 'deletion', 'word': 'а', 'context': 'а что если'},
        {'time': '9:24', 'type': 'insertion', 'word': 'НЕ', 'context': 'он не знал'},
        {'time': '15:24', 'type': 'insertion', 'word': 'то', 'context': 'то блюдочное'},
    ]


def auto_detect_related_files(json_file: str) -> dict:
    """
    Автоматически определяет связанные файлы на основе конвенции именования.

    По имени JSON файла (например, 01_filtered.json) ищет:
    - Аудио: 01_yandex.ogg в той же папке или Главы/
    - Оригинал: 01.docx в папке Главы/
    - Транскрипция: 01_transcript.json в той же папке

    Returns:
        dict с ключами 'audio', 'original', 'transcript' (или None для каждого)
    """
    result = {'audio': None, 'original': None, 'transcript': None}

    if not json_file:
        return result

    json_path = Path(json_file)

    # Определяем номер главы
    if HAS_CONFIG:
        chapter_id = FileNaming.get_chapter_id(json_path)
    else:
        match = re.search(r'(\d+)', json_path.stem)
        chapter_id = match.group(1) if match else None

    if not chapter_id:
        log.warning(f'Не удалось определить номер главы из {json_file}')
        return result

    log.info(f'Автоопределение файлов для главы {chapter_id}')

    # Папки для поиска
    json_dir = json_path.parent
    if HAS_CONFIG:
        chapters_dir = CHAPTERS_DIR
        audio_dir = AUDIO_DIR
        results_dir = RESULTS_DIR
        transcriptions_dir = TRANSCRIPTIONS_DIR
    else:
        project_dir = Path(__file__).parent.parent
        # Поддержка новой структуры Оригинал/Главы и Оригинал/Аудио
        original_dir = project_dir / 'Оригинал'
        chapters_dir = original_dir / 'Главы'
        audio_dir = original_dir / 'Аудио'
        results_dir = project_dir / 'Результаты проверки'
        transcriptions_dir = project_dir / 'Транскрибации'

    # Поиск аудио (новая структура: Оригинал/Аудио/)
    audio_patterns = [
        json_dir / f'{chapter_id}_yandex.ogg',
        json_dir / f'{chapter_id}.ogg',
        json_dir / f'{chapter_id}.mp3',
    ]
    if audio_dir:
        audio_patterns.extend([
            audio_dir / f'{chapter_id}.mp3',
            audio_dir / f'{chapter_id}.ogg',
            audio_dir / f'{chapter_id}_yandex.ogg',
        ])
    # Fallback: старая структура
    audio_patterns.extend([
        chapters_dir / f'{chapter_id}_yandex.ogg',
        chapters_dir / f'{chapter_id}.ogg',
        chapters_dir / f'{chapter_id}.mp3',
    ])
    for audio_path in audio_patterns:
        if audio_path.exists():
            result['audio'] = str(audio_path)
            log.info(f'  Аудио: {audio_path}')
            break

    # Поиск оригинала (новая структура: Оригинал/Главы/)
    # chapter_id может быть "01", "1" или "Глава2" — извлекаем число
    num_match = re.search(r'(\d+)', chapter_id)
    chapter_num = num_match.group(1).lstrip('0') or '1' if num_match else chapter_id  # "Глава2" -> "2", "01" -> "1"
    original_patterns = [
        chapters_dir / f'Глава {chapter_num}.docx',    # "Глава 1.docx"
        chapters_dir / f'Глава {chapter_id}.docx',     # "Глава 01.docx"
        chapters_dir / f'Глава{chapter_num}.docx',     # "Глава1.docx"
        chapters_dir / f'Глава{chapter_id}.docx',      # "Глава01.docx"
        chapters_dir / f'{chapter_id}.docx',           # "01.docx"
        json_dir / f'{chapter_id}.docx',
    ]
    for orig_path in original_patterns:
        if orig_path.exists():
            result['original'] = str(orig_path)
            log.info(f'  Оригинал: {orig_path}')
            break

    # Поиск транскрипции (новая структура: Транскрибации/Глава{N}/)
    # chapter_num_padded: "02" для двузначного номера
    chapter_num_padded = chapter_num.zfill(2)
    transcript_patterns = [
        json_dir / f'{chapter_id}_transcript.json',
        json_dir / f'{chapter_num_padded}_transcript.json',
        results_dir / f'{chapter_id}_transcript.json',
        results_dir / f'{chapter_num_padded}_transcript.json',
    ]
    if transcriptions_dir:
        # Пробуем разные форматы имени папки: "Глава1", "Глава01", "Глава 1", "Глава2"
        for folder_name in [f'Глава{chapter_num}', f'Глава{chapter_id}', f'Глава {chapter_num}', chapter_id]:
            chapter_trans_dir = transcriptions_dir / folder_name
            if chapter_trans_dir.exists():
                transcript_patterns.extend([
                    chapter_trans_dir / f'{chapter_id}_transcript_A_48kbps.json',
                    chapter_trans_dir / f'{chapter_id}_transcript.json',
                    chapter_trans_dir / f'{chapter_num_padded}_transcript_A_48kbps.json',
                    chapter_trans_dir / f'{chapter_num_padded}_transcript.json',
                    chapter_trans_dir / f'{chapter_num}_transcript_A_48kbps.json',
                    chapter_trans_dir / f'{chapter_num}_transcript.json',
                ])
    for trans_path in transcript_patterns:
        if trans_path.exists():
            result['transcript'] = str(trans_path)
            log.info(f'  Транскрипция: {trans_path}')
            break

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Веб-интерфейс для просмотра результатов проверки транскрипции'
    )
    parser.add_argument('json_file', nargs='?', help='JSON файл с результатами')
    parser.add_argument('--audio', '-a', help='Путь к аудиофайлу')
    parser.add_argument('--original', '-o', help='Оригинальный DOCX файл (для контекста с пунктуацией)')
    parser.add_argument('--transcript', '-t', help='JSON файл транскрипции (для позиции insertion)')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f'Порт сервера (по умолчанию {DEFAULT_PORT})')
    parser.add_argument('--demo', action='store_true', help='Запустить с демо-данными')
    parser.add_argument('--no-auto', action='store_true', help='Отключить автоопределение связанных файлов')

    args = parser.parse_args()

    # Автоопределение связанных файлов (если не отключено)
    auto_files = {'audio': None, 'original': None, 'transcript': None}
    if args.json_file and not args.no_auto:
        auto_files = auto_detect_related_files(args.json_file)
        if any(auto_files.values()):
            print("  Автоопределение файлов:")
            if auto_files['audio']:
                print(f"    Аудио: {auto_files['audio']}")
            if auto_files['original']:
                print(f"    Оригинал: {auto_files['original']}")
            if auto_files['transcript']:
                print(f"    Транскрипция: {auto_files['transcript']}")

    # Используем явно указанные файлы или автоопределённые
    audio_file = args.audio or auto_files['audio']
    original_file = args.original or auto_files['original']
    transcript_file = args.transcript or auto_files['transcript']

    # Загрузка данных
    if args.demo:
        errors = demo_errors()
    elif args.json_file:
        data = load_json_results(args.json_file)
        errors = data.get('errors', [])
    else:
        print("Запуск с демо-данными. Укажите JSON файл для реальных данных.")
        print("Использование: python web_viewer.py результат.json --audio аудио.mp3 --original оригинал.docx --transcript транскрипция.json")
        print()
        errors = demo_errors()

    # Загрузка транскрипции для insertion
    transcript_words = None
    if transcript_file:
        print(f"  Загрузка транскрипции: {transcript_file}")
        transcript_words = load_transcript_words(transcript_file)
        if transcript_words:
            print(f"  ✓ Загружено {len(transcript_words)} слов из транскрипции")

    # Обогащение контекстов из оригинального файла
    if original_file:
        print(f"  Загрузка оригинала: {original_file}")
        original_text = load_original_text(original_file)
        if original_text:
            errors = enrich_errors_with_original(errors, original_text, transcript_words)
            print(f"  ✓ Контексты обогащены из оригинала")

    # Путь к аудио
    audio_path = audio_file
    audio_name = None
    if audio_path:
        audio_name = Path(audio_path).name  # Только имя файла для HTML

    # Генерация HTML
    html = generate_html(errors, audio_name)

    # Запуск сервера
    run_server(html, audio_file, args.port, args.json_file)


if __name__ == '__main__':
    main()
