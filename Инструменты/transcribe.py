#!/usr/bin/env python3
"""
Transcribe v3.0 - Модуль транскрибации через Яндекс SpeechKit

Отправляет аудиофайлы на распознавание речи через Яндекс SpeechKit API.
Поддерживает асинхронное распознавание длинных аудио (до 4 часов).

Требования к аудио (рекомендуется конвертировать через audio_converter.py):
- Формат: OggOpus (рекомендуется), MP3, LPCM
- Каналы: моно
- Частота: 48000 Hz (для LPCM)
- Максимальная длительность: 4 часа
- Максимальный размер: 1 ГБ

Использование:
    python transcribe.py аудио.ogg --output результат.json
    python transcribe.py аудио.mp3 --model deferred-general  # дешевле, но дольше
    python transcribe.py аудио.ogg --folder-id b1g... --bucket my-bucket

Модели:
    general          - стандартная (быстрее)
    deferred-general - отложенная (дешевле, до 24ч обработки)
"""

import argparse
import base64
import json
import os
import sys
import time
import random
import requests
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Optional, Dict, Any

# Импорт централизованной конфигурации
try:
    from config import (
        YandexCloudConfig, FileNaming, RESULTS_DIR,
        format_duration, get_audio_info, check_file_exists
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Fallback значения если config.py недоступен
if HAS_CONFIG:
    YANDEX_S3_ENDPOINT = YandexCloudConfig.S3_ENDPOINT
    YANDEX_STT_SHORT_URL = YandexCloudConfig.STT_SHORT_URL
    YANDEX_STT_LONG_URL = YandexCloudConfig.STT_LONG_URL
    YANDEX_OPERATION_URL = YandexCloudConfig.OPERATION_URL
    AUDIO_ENCODINGS = YandexCloudConfig.AUDIO_ENCODINGS
    DEFAULT_MAX_RETRIES = YandexCloudConfig.MAX_RETRIES
    DEFAULT_BASE_DELAY = YandexCloudConfig.BASE_DELAY
    DEFAULT_MAX_DELAY = YandexCloudConfig.MAX_DELAY
    RETRYABLE_STATUS_CODES = YandexCloudConfig.RETRYABLE_STATUS_CODES
else:
    # Fallback константы
    YANDEX_S3_ENDPOINT = 'https://storage.yandexcloud.net'
    YANDEX_STT_SHORT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    YANDEX_STT_LONG_URL = "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"
    YANDEX_OPERATION_URL = "https://operation.api.cloud.yandex.net/operations/"
    AUDIO_ENCODINGS = {
        '.ogg': 'OGG_OPUS', '.opus': 'OGG_OPUS', '.mp3': 'MP3',
        '.raw': 'LINEAR16_PCM', '.pcm': 'LINEAR16_PCM', '.wav': 'LINEAR16_PCM',
    }
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0
    DEFAULT_MAX_DELAY = 30.0
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Ограничитель скорости запросов к API.

    Предотвращает 429-ошибки при массовой обработке глав.
    Использует алгоритм token bucket для контроля скорости.
    """

    def __init__(self, max_requests_per_second: float = 1.0, burst: int = 3):
        """
        Args:
            max_requests_per_second: максимальное среднее число запросов в секунду
            burst: максимальный "всплеск" (одновременных запросов)
        """
        self._rate = max_requests_per_second
        self._burst = burst
        self._tokens = float(burst)
        self._last_time = time.monotonic()
        self._lock_time = 0.0  # Время блокировки после 429

    def acquire(self) -> None:
        """Ожидает разрешения на следующий запрос."""
        while True:
            now = time.monotonic()

            # Если мы в режиме блокировки после 429
            if now < self._lock_time:
                wait = self._lock_time - now
                print(f"  ⏳ Rate limit: ожидание {wait:.1f}с...")
                time.sleep(wait)
                continue

            # Добавляем токены за прошедшее время
            elapsed = now - self._last_time
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_time = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Ожидаем пока появится токен
            wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def on_rate_limited(self, retry_after: float = 30.0) -> None:
        """Вызывается при получении 429 от API."""
        self._lock_time = time.monotonic() + retry_after
        self._tokens = 0.0
        print(f"  ⚠ Rate limited! Пауза {retry_after:.0f}с...")


# Глобальный rate limiter
_rate_limiter = RateLimiter(max_requests_per_second=1.0, burst=3)


def get_rate_limiter() -> RateLimiter:
    """Возвращает глобальный rate limiter."""
    return _rate_limiter


# =============================================================================
# RETRY-ЛОГИКА С EXPONENTIAL BACKOFF
# =============================================================================

class APIError(Exception):
    """Ошибка API с информацией для retry"""
    def __init__(self, message: str, status_code: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def calculate_backoff(attempt: int, base_delay: float = DEFAULT_BASE_DELAY,
                      max_delay: float = DEFAULT_MAX_DELAY) -> float:
    """
    Вычисляет задержку для retry с exponential backoff и jitter.

    Args:
        attempt: номер попытки (0-indexed)
        base_delay: базовая задержка в секундах
        max_delay: максимальная задержка в секундах

    Returns:
        Задержка в секундах с добавлением случайного jitter
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = base_delay * (2 ** attempt)

    # Добавляем jitter (±25%) для предотвращения thundering herd
    jitter = delay * 0.25 * (random.random() * 2 - 1)
    delay += jitter

    # Ограничиваем максимальной задержкой
    return min(delay, max_delay)


def is_retryable_error(status_code: int) -> bool:
    """Проверяет, можно ли повторить запрос при данном статусе"""
    return status_code in RETRYABLE_STATUS_CODES


def retry_request(func, *args, max_retries: int = DEFAULT_MAX_RETRIES,
                  base_delay: float = DEFAULT_BASE_DELAY, **kwargs) -> Any:
    """
    Выполняет функцию с retry-логикой и rate limiting.

    Args:
        func: функция для выполнения
        max_retries: максимальное число попыток
        base_delay: базовая задержка между попытками
        *args, **kwargs: аргументы для функции

    Returns:
        Результат функции

    Raises:
        APIError: если все попытки исчерпаны
    """
    last_error = None
    limiter = get_rate_limiter()

    for attempt in range(max_retries):
        try:
            # Rate limiting: ждём разрешения перед запросом
            limiter.acquire()
            return func(*args, **kwargs)

        except APIError as e:
            last_error = e

            # При 429 уведомляем rate limiter
            if e.status_code == 429:
                limiter.on_rate_limited(retry_after=30.0)

            if not e.retryable or attempt >= max_retries - 1:
                raise

            delay = calculate_backoff(attempt, base_delay)
            print(f"  ⚠ Ошибка API (код {e.status_code}), повтор через {delay:.1f}с...")
            print(f"    Попытка {attempt + 1}/{max_retries}")
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            last_error = e

            if attempt >= max_retries - 1:
                raise APIError(f"Сетевая ошибка: {e}", retryable=False)

            delay = calculate_backoff(attempt, base_delay)
            print(f"  ⚠ Сетевая ошибка, повтор через {delay:.1f}с...")
            print(f"    Попытка {attempt + 1}/{max_retries}")
            time.sleep(delay)

    raise last_error or APIError("Неизвестная ошибка", retryable=False)


def get_api_key():
    """Получает API ключ из переменных окружения или файла."""
    if HAS_CONFIG:
        return YandexCloudConfig.get_api_key()

    # Fallback логика
    api_key = os.environ.get('YANDEX_API_KEY')
    if api_key:
        return api_key

    key_file = Path.home() / '.yandex_api_key'
    if key_file.exists():
        return key_file.read_text().strip()

    project_key = Path(__file__).parent.parent / 'api_key.txt'
    if project_key.exists():
        return project_key.read_text().strip()

    keys_file = Path(__file__).parent.parent / 'api_keys.json'
    if keys_file.exists():
        try:
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys = json.load(f)
                return keys.get('speechkit', {}).get('secret')
        except:
            pass

    return None


def get_audio_encoding(file_path):
    """Определяет audioEncoding по расширению файла."""
    if HAS_CONFIG:
        return YandexCloudConfig.get_audio_encoding(file_path)
    ext = Path(file_path).suffix.lower()
    return AUDIO_ENCODINGS.get(ext, 'OGG_OPUS')


def _get_audio_info_fallback(audio_path):
    """Fallback версия get_audio_info если config.py недоступен."""
    info = {
        'duration': -1,
        'size_mb': os.path.getsize(audio_path) / 1024 / 1024,
        'encoding': get_audio_encoding(audio_path),
    }

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        info['duration'] = len(audio) / 1000.0
        info['channels'] = audio.channels
        info['sample_rate'] = audio.frame_rate
    except ImportError:
        pass
    except Exception as e:
        print(f"  Предупреждение: не удалось прочитать аудио: {e}")

    if info['duration'] < 0:
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(audio_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                info['duration'] = float(data.get('format', {}).get('duration', -1))
        except:
            pass

    return info


def _format_duration_fallback(seconds):
    """Fallback версия format_duration если config.py недоступен."""
    if seconds < 0:
        return "неизвестно"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}ч {minutes:02d}м {secs:02d}с"
    return f"{minutes}м {secs:02d}с"


# Используем функции из config.py или fallback
if not HAS_CONFIG:
    get_audio_info = _get_audio_info_fallback
    format_duration = _format_duration_fallback


def transcribe_short(audio_path, api_key, language='ru-RU'):
    """
    Синхронное распознавание короткого аудио (до 30 сек, до 1 МБ).
    """
    headers = {
        'Authorization': f'Api-Key {api_key}',
    }

    # Определяем формат
    ext = Path(audio_path).suffix.lower()
    if ext in ('.ogg', '.opus'):
        params = {'lang': language, 'format': 'oggopus'}
    elif ext == '.mp3':
        # Для MP3 используем oggopus после конвертации или отправляем как есть
        params = {'lang': language, 'format': 'oggopus'}
    else:
        params = {'lang': language, 'format': 'lpcm', 'sampleRateHertz': 48000}

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    def make_request():
        response = requests.post(
            YANDEX_STT_SHORT_URL,
            headers=headers,
            params=params,
            data=audio_data,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            retryable = is_retryable_error(response.status_code)
            raise APIError(
                f"Ошибка API: {response.status_code} - {response.text}",
                status_code=response.status_code,
                retryable=retryable
            )

    return retry_request(make_request)


def transcribe_long_with_base64(audio_path, api_key, language='ru-RU',
                                 model='general', folder_id=None):
    """
    Асинхронное распознавание длинного аудио.
    Отправляет файл как base64 в теле запроса (без Object Storage).

    Ограничение: файл должен быть < ~50 МБ после base64 кодирования.
    Для больших файлов нужен Object Storage.
    """
    headers = {
        'Authorization': f'Api-Key {api_key}',
        'Content-Type': 'application/json',
    }

    # Читаем и кодируем аудио
    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # Определяем формат
    encoding = get_audio_encoding(audio_path)

    # Формируем запрос
    specification = {
        'languageCode': language,
        'model': model,
        'profanityFilter': False,
        'audioEncoding': encoding,
        'rawResults': True,  # Получить таймкоды слов
    }

    # Для LPCM нужно указать частоту и каналы
    if encoding == 'LINEAR16_PCM':
        specification['sampleRateHertz'] = 48000
        specification['audioChannelCount'] = 1

    request_body = {
        'config': {
            'specification': specification
        },
        'audio': {
            'content': audio_base64
        }
    }

    if folder_id:
        specification['folderId'] = folder_id

    print(f"  Отправка на распознавание ({encoding})...")
    print(f"  Модель: {model}")

    def make_request():
        response = requests.post(
            YANDEX_STT_LONG_URL,
            headers=headers,
            json=request_body,
            timeout=120
        )

        if response.status_code != 200:
            retryable = is_retryable_error(response.status_code)
            raise APIError(
                f"Ошибка API: {response.status_code} - {response.text}",
                status_code=response.status_code,
                retryable=retryable
            )

        return response.json()

    operation = retry_request(make_request)
    operation_id = operation.get('id')

    if not operation_id:
        raise APIError(f"Не получен ID операции: {operation}", retryable=False)

    print(f"  Операция: {operation_id}")

    # Ожидаем завершения
    return wait_for_operation(operation_id, api_key)


def upload_to_storage(audio_path, bucket, object_key=None):
    """
    Загружает файл в Yandex Object Storage.

    Args:
        audio_path: путь к локальному файлу
        bucket: имя бакета
        object_key: путь в бакете (по умолчанию = имя файла)

    Returns:
        object_key загруженного файла
    """
    if object_key is None:
        object_key = Path(audio_path).name

    # Получаем credentials
    if HAS_CONFIG:
        access_key, secret_key = YandexCloudConfig.get_s3_credentials()
    else:
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    # Создаём S3 клиент для Yandex Object Storage
    s3_kwargs = {
        'endpoint_url': YANDEX_S3_ENDPOINT,
        'config': Config(signature_version='s3v4')
    }

    # Если есть ключи из config, используем их напрямую
    if access_key and secret_key:
        s3_kwargs['aws_access_key_id'] = access_key
        s3_kwargs['aws_secret_access_key'] = secret_key

    s3 = boto3.client('s3', **s3_kwargs)

    print(f"  Загрузка в Object Storage: s3://{bucket}/{object_key}")

    with open(audio_path, 'rb') as f:
        s3.upload_fileobj(f, bucket, object_key)

    print(f"  ✓ Загружено")
    return object_key


def transcribe_long_with_storage(audio_path, api_key, bucket, object_key,
                                  language='ru-RU', model='general', folder_id=None):
    """
    Асинхронное распознавание с файлом в Yandex Object Storage.
    Для файлов > 3 МБ (лимит gRPC для base64).

    Args:
        bucket: имя бакета в Object Storage
        object_key: путь к файлу в бакете
    """
    headers = {
        'Authorization': f'Api-Key {api_key}',
        'Content-Type': 'application/json',
    }

    encoding = get_audio_encoding(audio_path)

    specification = {
        'languageCode': language,
        'model': model,
        'profanityFilter': False,
        'audioEncoding': encoding,
        'rawResults': True,
    }

    if encoding == 'LINEAR16_PCM':
        specification['sampleRateHertz'] = 48000
        specification['audioChannelCount'] = 1

    # URI файла в Object Storage
    uri = f"https://storage.yandexcloud.net/{bucket}/{object_key}"

    request_body = {
        'config': {
            'specification': specification
        },
        'audio': {
            'uri': uri
        }
    }

    if folder_id:
        specification['folderId'] = folder_id

    print(f"  Файл в Object Storage: {uri}")
    print(f"  Модель: {model}")

    def make_request():
        response = requests.post(
            YANDEX_STT_LONG_URL,
            headers=headers,
            json=request_body,
            timeout=120
        )

        if response.status_code != 200:
            retryable = is_retryable_error(response.status_code)
            raise APIError(
                f"Ошибка API: {response.status_code} - {response.text}",
                status_code=response.status_code,
                retryable=retryable
            )

        return response.json()

    operation = retry_request(make_request)
    operation_id = operation.get('id')

    if not operation_id:
        raise APIError(f"Не получен ID операции: {operation}", retryable=False)

    print(f"  Операция: {operation_id}")

    return wait_for_operation(operation_id, api_key)


def wait_for_operation(operation_id, api_key, timeout=14400, poll_interval=15):
    """
    Ожидает завершения асинхронной операции.

    Args:
        timeout: максимальное время ожидания (по умолчанию 4 часа)
        poll_interval: интервал проверки в секундах
    """
    headers = {
        'Authorization': f'Api-Key {api_key}',
    }

    start_time = time.time()
    last_status = ""

    consecutive_errors = 0
    max_consecutive_errors = 5

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{YANDEX_OPERATION_URL}{operation_id}",
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                if is_retryable_error(response.status_code):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise APIError(
                            f"Слишком много ошибок проверки операции: {response.status_code}",
                            status_code=response.status_code,
                            retryable=False
                        )
                    delay = calculate_backoff(consecutive_errors - 1)
                    print(f"  ⚠ Ошибка проверки ({response.status_code}), повтор через {delay:.1f}с...")
                    time.sleep(delay)
                    continue
                else:
                    raise APIError(
                        f"Ошибка проверки операции: {response.status_code} - {response.text}",
                        status_code=response.status_code,
                        retryable=False
                    )

            consecutive_errors = 0  # Сброс счётчика при успехе
            operation = response.json()

        except requests.exceptions.RequestException as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise APIError(f"Сетевая ошибка при проверке операции: {e}", retryable=False)
            delay = calculate_backoff(consecutive_errors - 1)
            print(f"  ⚠ Сетевая ошибка, повтор через {delay:.1f}с...")
            time.sleep(delay)
            continue

        # Проверяем статус
        done = operation.get('done', False)
        metadata = operation.get('metadata', {})

        # Выводим прогресс если есть
        if metadata:
            progress = metadata.get('progressPercent', 0)
            status = f"  Прогресс: {progress}%"
            if status != last_status:
                print(status)
                last_status = status

        if done:
            if 'error' in operation:
                error = operation['error']
                raise APIError(
                    f"Ошибка распознавания: {error.get('message', error)}",
                    retryable=False
                )

            print("  ✓ Распознавание завершено")
            return operation.get('response', {})

        elapsed = int(time.time() - start_time)
        mins = elapsed // 60
        secs = elapsed % 60

        if elapsed % 60 == 0 and elapsed > 0:  # Каждую минуту
            print(f"  Ожидание... {mins}м {secs}с")

        time.sleep(poll_interval)

    raise APIError(f"Таймаут ожидания операции: {operation_id}", retryable=False)


def transcribe(audio_path, api_key=None, language='ru-RU', model='general',
               output_path=None, folder_id=None, bucket=None, object_key=None):
    """
    Главная функция транскрибации.

    Args:
        audio_path: путь к аудиофайлу
        api_key: API ключ (если не указан, ищет автоматически)
        language: язык распознавания (ru-RU, en-US, etc.)
        model: модель (general, deferred-general)
        output_path: путь для сохранения JSON
        folder_id: ID каталога в Yandex Cloud
        bucket: имя бакета Object Storage (для больших файлов)
        object_key: путь к файлу в бакете

    Returns:
        dict с результатами распознавания
    """
    if not api_key:
        api_key = get_api_key()

    if not api_key:
        raise Exception(
            "API ключ не найден. Укажите через:\n"
            "  --api-key параметр\n"
            "  YANDEX_API_KEY переменную окружения\n"
            "  ~/.yandex_api_key файл\n"
            "  api_keys.json в папке проекта"
        )

    # Получаем folder_id из config если не указан
    if not folder_id and HAS_CONFIG:
        folder_id = YandexCloudConfig.get_folder_id()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    # Получаем информацию о файле
    info = get_audio_info(str(audio_path))

    print(f"\n{'='*50}")
    print(f"  Файл: {audio_path.name}")
    print(f"  Размер: {info['size_mb']:.1f} MB")
    print(f"  Длительность: {format_duration(info['duration'])}")
    print(f"  Формат: {info['encoding']}")
    print(f"{'='*50}")

    # Проверяем ограничения
    if info['size_mb'] > 1024:
        raise Exception("Файл больше 1 ГБ. Максимальный размер для SpeechKit — 1 ГБ.")

    if info['duration'] > 0 and info['duration'] > 4 * 3600:
        raise Exception("Файл длиннее 4 часов. Максимальная длительность для SpeechKit — 4 часа.")

    # Выбираем метод распознавания
    if info['duration'] > 0 and info['duration'] <= 30 and info['size_mb'] <= 1:
        # Короткое синхронное распознавание
        print("\n  Метод: синхронное распознавание")
        result = transcribe_short(str(audio_path), api_key, language)

    elif info['size_mb'] < 3:
        # Base64 для небольших файлов (до 3 МБ из-за лимита gRPC)
        print("\n  Метод: асинхронное (base64)")
        result = transcribe_long_with_base64(
            str(audio_path), api_key, language, model, folder_id
        )

    else:
        # Файлы > 3 МБ — загружаем в Object Storage
        # Используем указанный бакет или из config.py
        if bucket:
            storage_bucket = bucket
        elif HAS_CONFIG:
            storage_bucket = YandexCloudConfig.get_bucket()
        else:
            storage_bucket = os.environ.get('YANDEX_BUCKET', 'audio-chapters-2026')

        storage_key = object_key or audio_path.name

        print(f"\n  Метод: асинхронное (Object Storage)")
        print(f"  Файл > 3 МБ, загрузка в бакет {storage_bucket}")

        # Загружаем файл в Object Storage
        upload_to_storage(str(audio_path), storage_bucket, storage_key)

        # Транскрибируем из Object Storage
        result = transcribe_long_with_storage(
            str(audio_path), api_key, storage_bucket, storage_key,
            language, model, folder_id
        )

    # Сохраняем результат
    if output_path:
        output_file = Path(output_path)
    else:
        # Используем FileNaming для правильного имени файла
        if HAS_CONFIG:
            chapter_id = FileNaming.get_chapter_id(audio_path)
            output_file = audio_path.parent / FileNaming.build_filename(chapter_id, 'transcript')
        else:
            output_file = audio_path.with_suffix('.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  Результат: {output_file}")

    # Выводим статистику
    chunks = result.get('chunks', [])
    if chunks:
        total_words = sum(
            len(alt.get('words', []))
            for chunk in chunks
            for alt in chunk.get('alternatives', [])
        )
        print(f"  Слов распознано: {total_words}")

        # Первые слова
        first_text = ""
        for chunk in chunks[:1]:
            for alt in chunk.get('alternatives', [])[:1]:
                first_text = alt.get('text', '')[:100]
        if first_text:
            print(f"  Начало: {first_text}...")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Транскрибация аудио через Яндекс SpeechKit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Форматы аудио:
  OGG_OPUS  — рекомендуется (компактный, не требует параметров)
  MP3       — поддерживается
  LINEAR16_PCM — максимальное качество (требует указания частоты)

Модели:
  general          — стандартная обработка
  deferred-general — отложенная (дешевле, но до 24ч)

Примеры:
  python transcribe.py глава.ogg
  python transcribe.py глава.mp3 --model deferred-general
  python transcribe.py глава.ogg --bucket my-bucket --object-key audio/глава.ogg
        """
    )
    parser.add_argument('audio', help='Путь к аудиофайлу')
    parser.add_argument('--output', '-o', help='Путь для сохранения результата')
    parser.add_argument('--api-key', '-k', help='API ключ Яндекс')
    parser.add_argument('--language', '-l', default='ru-RU',
                        help='Язык распознавания (по умолчанию: ru-RU)')
    parser.add_argument('--model', '-m', default='general',
                        choices=['general', 'deferred-general'],
                        help='Модель распознавания')
    parser.add_argument('--folder-id', '-f', help='ID каталога Yandex Cloud')
    parser.add_argument('--bucket', '-b', help='Имя бакета Object Storage')
    parser.add_argument('--object-key', help='Путь к файлу в бакете')

    args = parser.parse_args()

    try:
        result = transcribe(
            args.audio,
            api_key=args.api_key,
            language=args.language,
            model=args.model,
            output_path=args.output,
            folder_id=args.folder_id,
            bucket=args.bucket,
            object_key=args.object_key
        )

        print("\n✓ Готово!")

    except Exception as e:
        print(f"\n✗ Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
