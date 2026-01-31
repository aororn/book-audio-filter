#!/usr/bin/env python3
"""
Централизованная конфигурация проекта Яндекс Спич

Содержит:
- Пути к папкам и файлам
- Конвенция именования файлов
- Настройки алгоритмов
- Утилиты для работы с файлами
- Система логирования
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import re
import logging
import sys


# =============================================================================
# ПУТИ
# =============================================================================

# Корневая папка проекта
PROJECT_DIR = Path(__file__).parent.parent

# Основные папки
DICTIONARIES_DIR = PROJECT_DIR / 'Словари'
TOOLS_DIR = PROJECT_DIR / 'Инструменты'
RESULTS_DIR = PROJECT_DIR / 'Результаты проверки'
TEMP_DIR = PROJECT_DIR / 'Темп'
TESTS_DIR = PROJECT_DIR / 'Тесты'
READER_DIR = PROJECT_DIR / 'Чтецу'
TRANSCRIPTIONS_DIR = PROJECT_DIR / 'Транскрибации'

# Оригинальные материалы книги
ORIGINAL_DIR = PROJECT_DIR / 'Оригинал'
CHAPTERS_DIR = ORIGINAL_DIR / 'Главы'    # DOCX файлы с текстом глав
AUDIO_DIR = ORIGINAL_DIR / 'Аудио'       # MP3 файлы аудиокниги

# Словари
NAMES_DICT = DICTIONARIES_DIR / 'Словарь_имён_персонажей.txt'
PROTECTED_WORDS = DICTIONARIES_DIR / 'защищенные_слова.txt'
READER_ERRORS = DICTIONARIES_DIR / 'ошибки_чтеца.json'
CONFIG_JSON = DICTIONARIES_DIR / 'config.json'

# Базы данных (v6.1 — унифицирован путь к БД)
# ВАЖНО: Единственный источник правды — Словари/false_positives.db
# НЕ использовать Темп/ для хранения БД!
FALSE_POSITIVES_DB = DICTIONARIES_DIR / 'false_positives.db'
MORPH_CACHE_DB = TEMP_DIR / 'cache' / 'morph_cache.db'
ML_MODEL_DIR = TEMP_DIR / 'ml'


# =============================================================================
# КОНВЕНЦИЯ ИМЕНОВАНИЯ ФАЙЛОВ
# =============================================================================

class FileNaming:
    """
    Унифицированная конвенция именования файлов.

    Формат: {chapter}_{stage}.{ext}

    Где:
    - chapter: номер или название главы (01, 02, Глава_1)
    - stage: этап обработки (transcript, compared, filtered, final)

    Примеры:
    - 01_transcript.json    - транскрипция от Яндекса
    - 01_compared.json      - после сравнения
    - 01_filtered.json      - после фильтрации
    - 01_для_чтеца.docx     - финальный отчёт
    """

    # Этапы обработки
    STAGES = {
        'audio': 'yandex.ogg',           # Конвертированное аудио
        'transcript': 'transcript.json',  # Транскрипция Яндекса
        'normalized': 'normalized.txt',   # Нормализованный текст
        'compared': 'compared.json',      # После сравнения
        'filtered': 'filtered.json',      # После фильтрации
        'docx': 'для_чтеца.docx',         # Отчёт для чтеца
        'names': 'extracted_names.txt',   # Извлечённые имена
    }

    # Устаревшие имена файлов (для очистки)
    DEPRECATED = {
        '_final.json',      # Заменено на _filtered.json
        '_raw.json',        # Промежуточный файл
        '_transcript_compared.json',  # Неправильное именование
    }

    @classmethod
    def get_chapter_id(cls, path: Path) -> str:
        """Извлекает ID главы из пути к файлу."""
        stem = path.stem
        # Попытка извлечь номер (01, 02, ...)
        match = re.match(r'^(\d+)', stem)
        if match:
            return match.group(1)
        # Или имя типа "Глава_1"
        match = re.match(r'^(Глава[_\s]?\d+)', stem, re.IGNORECASE)
        if match:
            return match.group(1).replace(' ', '_')
        return stem

    @classmethod
    def build_filename(cls, chapter_id: str, stage: str) -> str:
        """Создаёт имя файла по конвенции."""
        if stage not in cls.STAGES:
            raise ValueError(f"Неизвестный этап: {stage}. Доступные: {list(cls.STAGES.keys())}")
        suffix = cls.STAGES[stage]
        return f"{chapter_id}_{suffix}"

    @classmethod
    def get_output_path(cls, chapter_id: str, stage: str, output_dir: Path) -> Path:
        """Возвращает полный путь к файлу результата."""
        filename = cls.build_filename(chapter_id, stage)
        return output_dir / filename

    @classmethod
    def is_deprecated(cls, path: Path) -> bool:
        """Проверяет, является ли файл устаревшим."""
        name = path.name
        for deprecated in cls.DEPRECATED:
            if deprecated in name:
                return True
        return False

    @classmethod
    def normalize_filename(cls, filename: str) -> str:
        """Нормализует имя файла (убирает пробелы, лишние символы)."""
        # Заменяем пробелы на подчёркивания
        filename = filename.replace(' ', '_')
        # Убираем двойные подчёркивания
        while '__' in filename:
            filename = filename.replace('__', '_')
        return filename

    # =================================================================
    # ПОИСК ФАЙЛОВ ПО ГЛАВАМ
    # =================================================================

    @classmethod
    def get_transcription_dir(cls, chapter_id: str) -> Path:
        """
        Возвращает папку для транскрипций главы.

        Правило хранения транскрипций:
            Транскрибации/Глава{N}/{chapter_id}_transcript.json

        Где N извлекается из chapter_id (01 → 1, 02 → 2).
        """
        chapter_num = re.sub(r'^0+', '', chapter_id) or '0'
        folder_name = f'Глава{chapter_num}'
        trans_dir = TRANSCRIPTIONS_DIR / folder_name
        trans_dir.mkdir(parents=True, exist_ok=True)
        return trans_dir

    @classmethod
    def get_transcription_path(cls, chapter_id: str) -> Path:
        """
        Возвращает путь к основному файлу транскрипции главы.

        Правило: Транскрибации/Глава{N}/{chapter_id}_transcript.json
        Пример:  Транскрибации/Глава1/01_transcript.json
        """
        trans_dir = cls.get_transcription_dir(chapter_id)
        filename = cls.build_filename(chapter_id, 'transcript')
        return trans_dir / filename

    @classmethod
    def find_transcription(cls, chapter_id: str) -> Optional[Path]:
        """
        Ищет файл транскрипции главы в стандартных местах.

        Порядок поиска:
        1. Транскрибации/Глава{N}/{chapter_id}_transcript.json (стандарт)
        2. Транскрибации/Глава{N}/*_transcript*.json (варианты)
        3. Результаты проверки/{chapter_id}/{chapter_id}_transcript.json
        """
        # 1. Стандартный путь
        standard = cls.get_transcription_path(chapter_id)
        if standard.exists():
            return standard

        # 2. Поиск вариантов в папке главы
        trans_dir = cls.get_transcription_dir(chapter_id)
        if trans_dir.exists():
            candidates = sorted(trans_dir.glob(f'{chapter_id}_transcript*.json'))
            if candidates:
                return candidates[0]

        # 3. Папка результатов
        results_path = RESULTS_DIR / chapter_id / cls.build_filename(chapter_id, 'transcript')
        if results_path.exists():
            return results_path

        return None

    @classmethod
    def get_original_path(cls, chapter_id: str) -> Optional[Path]:
        """
        Ищет оригинальный DOCX файл главы.

        Порядок поиска:
        1. Оригинал/Главы/Глава {N}.docx  (с пробелом)
        2. Оригинал/Главы/Глава{N}.docx   (без пробела)
        3. Оригинал/Главы/{chapter_id}.docx
        """
        chapter_num = re.sub(r'^0+', '', chapter_id) or '0'

        candidates = [
            CHAPTERS_DIR / f'Глава {chapter_num}.docx',
            CHAPTERS_DIR / f'Глава{chapter_num}.docx',
            CHAPTERS_DIR / f'{chapter_id}.docx',
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    @classmethod
    def get_audio_path(cls, chapter_id: str) -> Optional[Path]:
        """
        Ищет аудиофайл главы.

        Порядок поиска:
        1. Оригинал/Аудио/{chapter_id}.mp3
        2. Оригинал/Аудио/{chapter_id}_yandex.ogg (конвертированный)
        """
        candidates = [
            AUDIO_DIR / f'{chapter_id}.mp3',
            AUDIO_DIR / f'{chapter_id}_yandex.ogg',
        ]

        for path in candidates:
            if path.exists():
                return path

        return None


# =============================================================================
# ПРАВИЛА ХРАНЕНИЯ ФАЙЛОВ
# =============================================================================
#
# Все файлы проекта организованы по следующей структуре:
#
# Оригинал/
#   Главы/           - DOCX файлы оригинального текста (Глава 1.docx, Глава2.docx)
#   Аудио/            - MP3 файлы аудиокниги (01.mp3, 02.mp3)
#
# Транскрибации/
#   Глава{N}/          - Папка транскрипций для каждой главы
#     {NN}_transcript.json            - Основная транскрипция
#     {NN}_transcript_{variant}.json  - Варианты (разный битрейт, повторы)
#
# Результаты проверки/
#   {NN}/              - Папка результатов для каждой главы (ЕДИНСТВЕННОЕ место для compared/filtered)
#     {NN}_compared.json   - После сравнения (smart_compare)
#     {NN}_filtered.json   - После фильтрации (golden_filter) — ФИНАЛЬНЫЙ отчёт
#     {NN}_normalized.txt  - Нормализованный текст
#     {NN}_для_чтеца.docx - Отчёт для чтеца
#     extracted_names.txt  - Извлечённые имена
#
#   ВАЖНО: compared и filtered файлы хранятся ТОЛЬКО в Результаты проверки/{NN}/.
#   Не дублировать их в Темп/, корне проекта или других папках.
#   При повторном запуске пайплайна файлы перезаписываются (с --force).
#
# Чтецу/
#   {NN}_для_чтеца.docx   - Копия финального отчёта для чтеца (опционально)
#
# Словари/             - Словари и справочники (имена, омофоны, ошибки)
# Темп/                - Временные файлы (кэш, логи). НЕ хранить тут результаты!
#
# Тесты/               - Тестовые данные и golden standard
#   золотой_стандарт_глава{N}.json  - Эталонные ошибки
#   test_golden_standard.py         - Тест одной главы
#   test_golden_filter_unit.py      - Unit-тесты фильтра
#   run_full_test.py                - Полный тест обеих глав
#
# =============================================================================
# ПРАВИЛА СОЗДАНИЯ ЗОЛОТЫХ СТАНДАРТОВ
# =============================================================================
#
# Золотой стандарт — эталонный набор реальных ошибок чтеца для регрессионного
# тестирования фильтра после каждой доработки.
#
# ФОРМАТ ФАЙЛА: золотой_стандарт_глава{N}.json
#
#   {
#     "_comment": "Золотой стандарт ошибок главы N. Эти ошибки ДОЛЖНЫ находиться при любой проверке.",
#     "chapter": "Глава N",
#     "audio": "{NN}.mp3",
#     "errors": [
#       {
#         "time": "M:SS",              // время ошибки (человекочитаемое)
#         "time_seconds": 123,          // время ошибки в секундах
#         "type": "substitution",       // тип: substitution | insertion | deletion
#         "wrong": "услышанное",        // что услышал Яндекс (для insertion — лишнее слово)
#         "correct": "правильное",      // что должно быть (для deletion — пропущенное)
#         "context": "...фрагмент оригинала с правильным словом..."
#       }
#     ]
#   }
#
# ОБЯЗАТЕЛЬНЫЕ ПОЛЯ ОШИБКИ:
#   - time          (str)  — формат "M:SS" или "H:MM:SS"
#   - time_seconds  (int)  — время в секундах
#   - type          (str)  — substitution, insertion или deletion
#   - wrong         (str)  — что услышал Яндекс ("" для deletion)
#   - correct       (str)  — что должно быть ("" для insertion)
#   - context       (str)  — фрагмент оригинала для контекста
#
# ПРАВИЛА СОЗДАНИЯ:
#   1. Каждая ошибка ВЕРИФИЦИРОВАНА на слух — прослушана в аудиозаписи
#   2. Ошибки включают ТОЛЬКО реальные ошибки чтеца, не ошибки Яндекса
#   3. Формат полей ЕДИНЫЙ для всех глав (не использовать spoken/expected, timestamp, SUB/INS/DEL)
#   4. Файл хранится в Тесты/золотой_стандарт_глава{N}.json
#   5. При добавлении новой главы — создать стандарт ДО первого запуска фильтра
#   6. После доработки фильтра — запустить run_full_test.py для проверки регрессий
#
# ИМЕНОВАНИЕ:
#   золотой_стандарт_глава1.json   — глава 1
#   золотой_стандарт_глава2.json   — глава 2
#   золотой_стандарт_глава{N}.json — глава N
#
# ЗАПУСК ТЕСТОВ:
#   python Тесты/run_full_test.py              — обе главы (пайплайн + тест)
#   python Тесты/run_full_test.py --chapter 1  — только глава 1
#   python Тесты/run_full_test.py --skip-pipeline — только золотой тест


# =============================================================================
# НАСТРОЙКИ АЛГОРИТМОВ
# =============================================================================

def _load_algorithm_config() -> dict:
    """Загружает конфигурацию алгоритмов из Словари/config.json."""
    import json
    if CONFIG_JSON.exists():
        try:
            with open(CONFIG_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_algo_config = _load_algorithm_config()


class SmartCompareConfig:
    """Настройки умного сравнения. Загружаются из Словари/config.json."""
    _sc = _algo_config.get('smart_compare', {})
    THRESHOLD = _sc.get('threshold', 0.7)
    PHANTOM_SECONDS = _sc.get('phantom_seconds', -1)
    MIN_WORD_LEN_ANCHOR = _sc.get('min_word_len_anchor', 4)


class GoldenFilterConfig:
    """Настройки фильтра отсева. Загружаются из Словари/config.json."""
    _gf = _algo_config.get('golden_filter', {})
    LEVENSHTEIN_THRESHOLD = _gf.get('levenshtein_threshold', 2)
    USE_LEMMATIZATION = _gf.get('use_lemmatization', True)
    USE_HOMOPHONES = _gf.get('use_homophones', True)
    USE_PHONETIC = _gf.get('use_phonetic', True)


class YandexCloudConfig:
    """
    Настройки Yandex Cloud (SpeechKit + Object Storage).

    Ключи загружаются из api_keys.json в корне проекта.
    """
    # Endpoints
    S3_ENDPOINT = 'https://storage.yandexcloud.net'
    STT_SHORT_URL = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize'
    STT_LONG_URL = 'https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize'
    OPERATION_URL = 'https://operation.api.cloud.yandex.net/operations/'

    # Файл с ключами
    KEYS_FILE = PROJECT_DIR / 'api_keys.json'

    # Retry конфигурация
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # секунды
    MAX_DELAY = 30.0  # секунды
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    # Поддерживаемые форматы аудио
    AUDIO_ENCODINGS = {
        '.ogg': 'OGG_OPUS',
        '.opus': 'OGG_OPUS',
        '.mp3': 'MP3',
        '.raw': 'LINEAR16_PCM',
        '.pcm': 'LINEAR16_PCM',
        '.wav': 'LINEAR16_PCM',
    }

    # Кэш для ключей
    _keys_cache = None

    @classmethod
    def load_keys(cls) -> dict:
        """Загружает ключи из api_keys.json."""
        if cls._keys_cache is not None:
            return cls._keys_cache

        if not cls.KEYS_FILE.exists():
            return {}

        import json
        try:
            with open(cls.KEYS_FILE, 'r', encoding='utf-8') as f:
                cls._keys_cache = json.load(f)
                return cls._keys_cache
        except Exception:
            return {}

    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Получает API ключ SpeechKit."""
        import os

        # 1. Переменная окружения
        api_key = os.environ.get('YANDEX_API_KEY')
        if api_key:
            return api_key

        # 2. api_keys.json
        keys = cls.load_keys()
        api_key = keys.get('speechkit', {}).get('secret')
        if api_key:
            return api_key

        # 3. ~/.yandex_api_key
        key_file = Path.home() / '.yandex_api_key'
        if key_file.exists():
            return key_file.read_text().strip()

        # 4. api_key.txt в проекте
        project_key = PROJECT_DIR / 'api_key.txt'
        if project_key.exists():
            return project_key.read_text().strip()

        return None

    @classmethod
    def get_s3_credentials(cls) -> tuple:
        """Возвращает (access_key_id, secret_key) для Object Storage."""
        import os

        # Переменные окружения
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if access_key and secret_key:
            return access_key, secret_key

        # api_keys.json
        keys = cls.load_keys()
        storage = keys.get('cloud_storage', {})
        return (
            storage.get('access_key_id'),
            storage.get('secret_key')
        )

    @classmethod
    def get_bucket(cls) -> str:
        """Возвращает имя бакета."""
        import os
        return os.environ.get('YANDEX_BUCKET') or cls.load_keys().get('bucket', 'audio-chapters-2026')

    @classmethod
    def get_folder_id(cls) -> Optional[str]:
        """Возвращает ID каталога Yandex Cloud."""
        import os
        return os.environ.get('YANDEX_FOLDER_ID') or cls.load_keys().get('folder_id')

    @classmethod
    def get_audio_encoding(cls, file_path) -> str:
        """Определяет audioEncoding по расширению файла."""
        ext = Path(file_path).suffix.lower()
        return cls.AUDIO_ENCODINGS.get(ext, 'OGG_OPUS')


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def ensure_dirs_exist():
    """Создаёт необходимые папки если их нет."""
    for dir_path in [RESULTS_DIR, TEMP_DIR, READER_DIR, TRANSCRIPTIONS_DIR, ORIGINAL_DIR, CHAPTERS_DIR, AUDIO_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    """Форматирует длительность в читаемый вид."""
    if seconds < 0:
        return "неизвестно"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}ч {minutes:02d}м {secs:02d}с"
    return f"{minutes}м {secs:02d}с"


def get_audio_info(audio_path) -> dict:
    """
    Получает информацию об аудиофайле.

    Returns:
        dict с ключами: duration, size_mb, encoding, channels, sample_rate
    """
    import os
    import json
    import subprocess

    audio_path = Path(audio_path)
    info = {
        'duration': -1,
        'size_mb': os.path.getsize(audio_path) / 1024 / 1024,
        'encoding': YandexCloudConfig.get_audio_encoding(audio_path),
        'channels': None,
        'sample_rate': None,
    }

    # Пробуем получить информацию через pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        info['duration'] = len(audio) / 1000.0
        info['channels'] = audio.channels
        info['sample_rate'] = audio.frame_rate
        return info
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: ffprobe
    if info['duration'] < 0:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                str(audio_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                fmt = data.get('format', {})
                info['duration'] = float(fmt.get('duration', -1))

                # Ищем аудио поток
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        info['channels'] = stream.get('channels')
                        info['sample_rate'] = int(stream.get('sample_rate', 0)) or None
                        break
        except Exception:
            pass

    return info


def check_file_exists(path: Path, action: str = 'skip') -> bool:
    """
    Проверяет существование файла и решает что делать.

    Args:
        path: путь к файлу
        action: 'skip' - пропустить если существует,
                'overwrite' - перезаписать,
                'ask' - вывести предупреждение

    Returns:
        True если можно продолжать (файл не существует или action='overwrite')
    """
    if not path.exists():
        return True

    if action == 'overwrite':
        return True

    if action == 'ask':
        print(f"  ⚠ Файл уже существует: {path.name}")
        return True  # Продолжаем, но предупредили

    # action == 'skip'
    print(f"  → Файл уже существует, пропускаем: {path.name}")
    return False


def get_chapter_output_dir(chapter_id: str) -> Path:
    """Возвращает папку для результатов главы."""
    output_dir = RESULTS_DIR / chapter_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_file_correspondence(
    transcript_path: Path,
    text_path: Path,
    tolerance_words: int = 10
) -> Dict[str, Any]:
    """
    Проверяет соответствие транскрипции и текста.

    Args:
        transcript_path: Путь к JSON транскрипции
        text_path: Путь к оригинальному тексту
        tolerance_words: Допустимое число несовпадающих слов в начале

    Returns:
        Словарь с результатами проверки
    """
    import json

    result = {
        'valid': False,
        'transcript_words': 0,
        'text_words': 0,
        'matching_start_words': 0,
        'error': None
    }

    try:
        # Загружаем транскрипцию
        with open(transcript_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)

        # Извлекаем первые слова транскрипции
        trans_words = []
        for chunk in trans_data.get('chunks', [])[:10]:
            for alt in chunk.get('alternatives', []):
                for w in alt.get('words', []):
                    word = w.get('word', '').lower()
                    if word:
                        trans_words.append(word)

        # Загружаем текст
        if text_path.suffix.lower() == '.docx':
            from docx import Document
            doc = Document(str(text_path))
            text = ' '.join(p.text for p in doc.paragraphs)
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Нормализуем текст
        import re
        text = text.lower().replace('ё', 'е')
        text = re.sub(r'[^\w\s]', ' ', text)
        text_words = text.split()[:50]

        result['transcript_words'] = len(trans_words)
        result['text_words'] = len(text_words)

        # Сравниваем начала
        matching = 0
        for i, tw in enumerate(trans_words[:20]):
            for j in range(max(0, i - 5), min(len(text_words), i + 10)):
                if text_words[j] == tw or (
                    len(tw) >= 4 and len(text_words[j]) >= 4 and
                    abs(len(tw) - len(text_words[j])) <= 1
                ):
                    matching += 1
                    break

        result['matching_start_words'] = matching
        result['valid'] = matching >= 10  # Минимум 10 совпадений из 20

    except Exception as e:
        result['error'] = str(e)

    return result


def cleanup_deprecated_files(output_dir: Path, dry_run: bool = True) -> list:
    """
    Очищает устаревшие файлы из папки результатов.

    Args:
        output_dir: Папка для очистки
        dry_run: Если True, только показывает что будет удалено

    Returns:
        Список удалённых (или найденных при dry_run) файлов
    """
    deprecated = []

    for file_path in output_dir.iterdir():
        if file_path.is_file() and FileNaming.is_deprecated(file_path):
            deprecated.append(file_path)
            if not dry_run:
                file_path.unlink()

    return deprecated


# =============================================================================
# СИСТЕМА ЛОГИРОВАНИЯ
# =============================================================================

# Папка для логов
LOGS_DIR = TEMP_DIR / 'logs'


class LogConfig:
    """Конфигурация системы логирования."""

    # Формат сообщений
    FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # Уровни логирования
    DEFAULT_LEVEL = 'INFO'
    FILE_LEVEL = 'DEBUG'  # В файл пишем всё

    # Ротация логов
    MAX_LOG_FILES = 10  # Хранить последние N логов
    MAX_LOG_SIZE_MB = 10  # Максимальный размер файла

    # Имя текущей сессии логирования
    _session_id = None
    _initialized = False


def setup_logging(
    level: str = None,
    log_file: str = None,
    module_name: str = None,
    console: bool = True,
    session_id: str = None
) -> logging.Logger:
    """
    Настраивает логирование для проекта.

    Args:
        level: Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Путь к файлу лога (None = авто в LOGS_DIR)
        module_name: Имя модуля для логгера
        console: Выводить в консоль
        session_id: ID сессии для имени файла

    Returns:
        Настроенный логгер

    Использование:
        from config import setup_logging, get_logger

        # В начале main():
        setup_logging(level='DEBUG')  # Настроить один раз

        # В модулях:
        logger = get_logger(__name__)
        logger.info("Сообщение")
        logger.error("Ошибка", exc_info=True)
    """
    # Уровень по умолчанию
    if level is None:
        level = LogConfig.DEFAULT_LEVEL

    # Преобразуем строку в уровень
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Создаём папку логов
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Генерируем ID сессии
    if session_id is None:
        if LogConfig._session_id is None:
            LogConfig._session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = LogConfig._session_id

    # Определяем файл лога
    if log_file is None:
        log_file = LOGS_DIR / f'pipeline_{session_id}.log'
    else:
        log_file = Path(log_file)

    # Форматтер
    formatter = logging.Formatter(
        LogConfig.FORMAT,
        datefmt=LogConfig.DATE_FORMAT
    )

    # Получаем корневой логгер проекта
    logger_name = module_name or 'yandex_speech'
    logger = logging.getLogger(logger_name)

    # Не настраиваем повторно
    if LogConfig._initialized and logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Логгер принимает всё, фильтруют handlers

    # Очищаем старые handlers (для повторных вызовов)
    logger.handlers.clear()

    # Handler для файла (DEBUG уровень — всё)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler для консоли (заданный уровень)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        # Упрощённый формат для консоли
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s' if log_level >= logging.WARNING
            else '%(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Ротация старых логов
    cleanup_old_logs(LOGS_DIR, LogConfig.MAX_LOG_FILES)

    LogConfig._initialized = True

    # Логируем начало сессии
    logger.info(f"=== Сессия логирования начата: {session_id} ===")
    logger.debug(f"Лог-файл: {log_file}")
    logger.debug(f"Уровень консоли: {level}")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Получает логгер для модуля.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Логгер

    Использование:
        from config import get_logger
        logger = get_logger(__name__)
        logger.info("Обработка файла...")
    """
    if name is None:
        name = 'yandex_speech'

    # Если имя начинается с пути, берём только имя модуля
    if '/' in name or '\\' in name:
        name = Path(name).stem

    # Создаём дочерний логгер
    logger = logging.getLogger(f'yandex_speech.{name}')

    return logger


def cleanup_old_logs(logs_dir: Path, keep_count: int = 10) -> int:
    """
    Удаляет старые лог-файлы, оставляя последние N.

    Args:
        logs_dir: Папка с логами
        keep_count: Сколько файлов оставить

    Returns:
        Количество удалённых файлов
    """
    if not logs_dir.exists():
        return 0

    log_files = sorted(
        logs_dir.glob('pipeline_*.log'),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    deleted = 0
    for old_log in log_files[keep_count:]:
        try:
            old_log.unlink()
            deleted += 1
        except OSError:
            pass

    return deleted


def log_exception(logger: logging.Logger, msg: str, exc: Exception = None):
    """
    Логирует исключение с полным traceback.

    Args:
        logger: Логгер
        msg: Сообщение
        exc: Исключение (если None, берёт текущее)
    """
    if exc:
        logger.error(f"{msg}: {exc}", exc_info=True)
    else:
        logger.error(msg, exc_info=True)


# =============================================================================
# ВЕРСИЯ
# =============================================================================

VERSION = '6.1.0'
VERSION_DATE = '2026-01-31'
