#!/usr/bin/env python3
"""
batch_transcribe_from_bucket.py v1.0

Пакетная транскрибация аудиофайлов из Yandex Object Storage.
Запускает транскрибацию всех файлов из бакета параллельно или последовательно.

Использование:
    python batch_transcribe_from_bucket.py              # Все файлы
    python batch_transcribe_from_bucket.py --list       # Только показать файлы
    python batch_transcribe_from_bucket.py --filter 01  # Только файлы с "01" в имени
    python batch_transcribe_from_bucket.py --parallel   # Параллельный режим (осторожно с лимитами API)
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Unbuffered print
print = partial(print, flush=True)

# Добавляем путь к инструментам
sys.path.insert(0, str(Path(__file__).parent))

import boto3
from botocore.config import Config
import requests

# Импортируем конфигурацию
try:
    from config import YandexCloudConfig, PROJECT_DIR, TRANSCRIPTIONS_DIR
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    PROJECT_DIR = Path(__file__).parent.parent
    TRANSCRIPTIONS_DIR = PROJECT_DIR / 'Транскрибации'

# Константы API
YANDEX_STT_LONG_URL = 'https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize'
YANDEX_OPERATION_URL = 'https://operation.api.cloud.yandex.net/operations/'
YANDEX_S3_ENDPOINT = 'https://storage.yandexcloud.net'


def load_api_keys():
    """Загружает API ключи из файла."""
    keys_path = PROJECT_DIR / 'api_keys.json'
    if not keys_path.exists():
        raise FileNotFoundError(f"Файл ключей не найден: {keys_path}")

    with open(keys_path) as f:
        return json.load(f)


def get_s3_client(keys):
    """Создаёт S3 клиент для Yandex Object Storage."""
    return boto3.client(
        's3',
        endpoint_url=YANDEX_S3_ENDPOINT,
        aws_access_key_id=keys['cloud_storage']['access_key_id'],
        aws_secret_access_key=keys['cloud_storage']['secret_key'],
        config=Config(signature_version='s3v4')
    )


def list_bucket_files(keys, filter_pattern=None):
    """Получает список файлов из бакета."""
    s3 = get_s3_client(keys)
    bucket = keys['bucket']

    response = s3.list_objects_v2(Bucket=bucket)
    files = []

    for obj in response.get('Contents', []):
        key = obj['Key']
        if filter_pattern and filter_pattern not in key:
            continue
        if key.endswith('.ogg'):
            files.append({
                'key': key,
                'size_mb': obj['Size'] / 1024 / 1024,
                'uri': f"https://storage.yandexcloud.net/{bucket}/{key}"
            })

    return files


def get_chapter_from_filename(filename):
    """Извлекает номер главы из имени файла."""
    # 01_yandex.ogg -> 01, 02_yandex.ogg -> 02
    base = filename.split('_')[0]
    if base.isdigit():
        return base
    return None


def get_output_path(filename):
    """Определяет путь для сохранения транскрипции."""
    chapter = get_chapter_from_filename(filename)
    if not chapter:
        return TRANSCRIPTIONS_DIR / f"{filename.replace('.ogg', '')}_transcript.json"

    # Определяем подпапку главы
    chapter_dir = TRANSCRIPTIONS_DIR / f"Глава{int(chapter)}"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    # Имя файла: 01_transcript_NEW_20260125.json
    base_name = filename.replace('.ogg', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    return chapter_dir / f"{base_name}_transcript_NEW_{timestamp}.json"


def transcribe_from_uri(uri, api_key, folder_id, output_path):
    """
    Запускает транскрибацию файла из Object Storage.

    Args:
        uri: URI файла в Object Storage (https://storage.yandexcloud.net/bucket/key)
        api_key: API ключ SpeechKit
        folder_id: ID каталога Yandex Cloud
        output_path: путь для сохранения результата

    Returns:
        dict с результатом или None при ошибке
    """
    headers = {
        'Authorization': f'Api-Key {api_key}',
        'Content-Type': 'application/json',
    }

    request_body = {
        'config': {
            'specification': {
                'languageCode': 'ru-RU',
                'model': 'general',
                'profanityFilter': False,
                'audioEncoding': 'OGG_OPUS',
                'rawResults': True,
                'folderId': folder_id,
            }
        },
        'audio': {
            'uri': uri
        }
    }

    filename = uri.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"  Файл: {filename}")
    print(f"  URI: {uri}")
    print(f"  Выход: {output_path.name}")
    print(f"{'='*60}")

    # Отправляем запрос
    try:
        response = requests.post(
            YANDEX_STT_LONG_URL,
            headers=headers,
            json=request_body,
            timeout=120
        )

        if response.status_code != 200:
            print(f"  ✗ Ошибка API: {response.status_code} - {response.text}")
            return None

        operation = response.json()
        operation_id = operation.get('id')

        if not operation_id:
            print(f"  ✗ Не получен ID операции: {operation}")
            return None

        print(f"  Операция: {operation_id}")

    except Exception as e:
        print(f"  ✗ Ошибка запроса: {e}")
        return None

    # Ждём завершения операции
    result = wait_for_operation(operation_id, api_key)

    if result:
        # Сохраняем результат
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Сохранено: {output_path}")

        # Статистика
        chunks = result.get('chunks', [])
        words = sum(
            len(chunk.get('alternatives', [{}])[0].get('words', []))
            for chunk in chunks
        )
        print(f"  Чанков: {len(chunks)}, слов: {words}")

    return result


def wait_for_operation(operation_id, api_key, timeout=14400, poll_interval=15):
    """Ожидает завершения асинхронной операции."""
    headers = {
        'Authorization': f'Api-Key {api_key}',
    }

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{YANDEX_OPERATION_URL}{operation_id}",
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"  ⚠ Ошибка проверки: {response.status_code}")
                time.sleep(poll_interval)
                continue

            operation = response.json()
            done = operation.get('done', False)

            # Прогресс
            metadata = operation.get('metadata', {})
            if metadata:
                progress = metadata.get('progressPercent', 0)
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                if elapsed % 30 == 0:
                    print(f"  Прогресс: {progress}% ({mins}м {secs}с)")

            if done:
                if 'error' in operation:
                    error = operation['error']
                    print(f"  ✗ Ошибка распознавания: {error}")
                    return None

                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                print(f"  ✓ Готово за {mins}м {secs}с")
                return operation.get('response', {})

        except Exception as e:
            print(f"  ⚠ Сетевая ошибка: {e}")

        time.sleep(poll_interval)

    print(f"  ✗ Таймаут операции")
    return None


def main():
    parser = argparse.ArgumentParser(description='Пакетная транскрибация из бакета')
    parser.add_argument('--list', action='store_true', help='Только показать файлы')
    parser.add_argument('--filter', '-f', help='Фильтр по имени файла')
    parser.add_argument('--parallel', action='store_true', help='Параллельный режим')
    parser.add_argument('--max-workers', type=int, default=3, help='Макс. параллельных задач')
    args = parser.parse_args()

    # Загружаем ключи
    keys = load_api_keys()
    api_key = keys['speechkit']['secret']
    folder_id = keys['folder_id']

    # Получаем список файлов
    files = list_bucket_files(keys, args.filter)

    print(f"\nФайлы в бакете ({len(files)}):")
    for f in files:
        print(f"  {f['key']} ({f['size_mb']:.1f} MB)")

    if args.list:
        return

    if not files:
        print("\nНет файлов для транскрибации")
        return

    print(f"\nНачинаю транскрибацию {len(files)} файлов...")

    results = {}

    if args.parallel:
        # Параллельный режим
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for f in files:
                output_path = get_output_path(f['key'])
                future = executor.submit(
                    transcribe_from_uri,
                    f['uri'], api_key, folder_id, output_path
                )
                futures[future] = f['key']

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    results[key] = 'ok' if result else 'error'
                except Exception as e:
                    print(f"  ✗ {key}: {e}")
                    results[key] = 'error'
    else:
        # Последовательный режим
        for f in files:
            output_path = get_output_path(f['key'])
            result = transcribe_from_uri(f['uri'], api_key, folder_id, output_path)
            results[f['key']] = 'ok' if result else 'error'

    # Итоги
    print(f"\n{'='*60}")
    print("ИТОГИ:")
    ok_count = sum(1 for v in results.values() if v == 'ok')
    print(f"  Успешно: {ok_count}/{len(files)}")
    for key, status in results.items():
        mark = '✓' if status == 'ok' else '✗'
        print(f"  {mark} {key}")


if __name__ == '__main__':
    main()
