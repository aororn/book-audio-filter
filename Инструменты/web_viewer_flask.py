#!/usr/bin/env python3
"""
Web Viewer Flask v1.0 — стабильная версия веб-просмотрщика ошибок

Использует Flask с правильной обработкой Range-запросов для аудио.
Решает проблему зависания плеера при паузе/перемотке.

Использование:
    python web_viewer_flask.py 05  # Глава 5
    python web_viewer_flask.py 01  # Глава 1
    python web_viewer_flask.py /путь/к/filtered.json --audio /путь/к/audio.ogg
"""

from flask import Flask, request, Response
from pathlib import Path
import argparse
import json
import os
import re
import webbrowser

VERSION = "1.0.0"

app = Flask(__name__)

# Глобальные переменные (устанавливаются при запуске)
RESULTS_DIR = None
TEMPLATES_DIR = None
JSON_FILE = None
AUDIO_FILE = None
CHAPTER_ID = None


def find_project_root():
    """Находит корень проекта"""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "Инструменты").exists() and (current / "Результаты проверки").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


@app.route('/')
def index():
    """Главная страница с ошибками"""
    template_path = TEMPLATES_DIR / "viewer.html"

    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    audio_name = Path(AUDIO_FILE).name if AUDIO_FILE else None
    inject = {
        'errors': data.get('errors', []),
        'audio': audio_name
    }
    html = html.replace('__ERROR_DATA__', json.dumps(inject, ensure_ascii=False))
    return html


@app.route('/<path:filename>')
def serve_audio(filename):
    """Отдача аудиофайла с поддержкой Range-запросов"""
    if not filename.endswith(('.ogg', '.mp3', '.wav')):
        return "Not found", 404

    # Определяем путь к аудио
    if AUDIO_FILE and Path(AUDIO_FILE).name == filename:
        audio_path = Path(AUDIO_FILE)
    else:
        audio_path = RESULTS_DIR / filename

    if not audio_path.exists():
        return "Audio not found", 404

    file_size = audio_path.stat().st_size
    mimetype = 'audio/ogg' if filename.endswith('.ogg') else 'audio/mpeg'

    range_header = request.headers.get('Range')

    if range_header:
        m = re.match(r'bytes=(\d*)-(\d*)', range_header)
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(audio_path, 'rb') as f:
                f.seek(start)
                data = f.read(length)

            resp = Response(
                data,
                status=206,
                mimetype=mimetype,
                direct_passthrough=True
            )
            resp.headers['Accept-Ranges'] = 'bytes'
            resp.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            resp.headers['Content-Length'] = str(length)
            resp.headers['Cache-Control'] = 'no-cache'
            return resp

    # Полный файл
    with open(audio_path, 'rb') as f:
        data = f.read()

    resp = Response(data, mimetype=mimetype, direct_passthrough=True)
    resp.headers['Accept-Ranges'] = 'bytes'
    resp.headers['Content-Length'] = str(file_size)
    return resp


def setup_paths(chapter_or_json: str, audio: str = None):
    """Настройка путей на основе аргументов"""
    global RESULTS_DIR, TEMPLATES_DIR, JSON_FILE, AUDIO_FILE, CHAPTER_ID

    project_root = find_project_root()
    TEMPLATES_DIR = project_root / "Инструменты" / "templates"

    # Если передан номер главы (01, 02, ...)
    if re.match(r'^\d{1,2}$', chapter_or_json):
        CHAPTER_ID = chapter_or_json.zfill(2)
        RESULTS_DIR = project_root / "Результаты проверки" / CHAPTER_ID
        JSON_FILE = RESULTS_DIR / f"{CHAPTER_ID}_filtered.json"
        AUDIO_FILE = RESULTS_DIR / f"{CHAPTER_ID}_yandex.ogg"
    else:
        # Передан путь к JSON
        JSON_FILE = Path(chapter_or_json)
        RESULTS_DIR = JSON_FILE.parent
        CHAPTER_ID = re.search(r'(\d{2})', JSON_FILE.name)
        CHAPTER_ID = CHAPTER_ID.group(1) if CHAPTER_ID else "01"

        if audio:
            AUDIO_FILE = Path(audio)
        else:
            # Автоопределение аудио
            AUDIO_FILE = RESULTS_DIR / f"{CHAPTER_ID}_yandex.ogg"

    # Проверки
    if not JSON_FILE.exists():
        raise FileNotFoundError(f"JSON не найден: {JSON_FILE}")
    if AUDIO_FILE and not AUDIO_FILE.exists():
        print(f"⚠ Аудио не найдено: {AUDIO_FILE}")
        AUDIO_FILE = None


def main():
    global RESULTS_DIR, TEMPLATES_DIR, JSON_FILE, AUDIO_FILE, CHAPTER_ID

    parser = argparse.ArgumentParser(description='Web Viewer Flask — просмотр ошибок')
    parser.add_argument('chapter', help='Номер главы (01, 02...) или путь к JSON')
    parser.add_argument('--audio', help='Путь к аудиофайлу')
    parser.add_argument('--port', type=int, default=5050, help='Порт (по умолчанию 5050)')
    parser.add_argument('--no-browser', action='store_true', help='Не открывать браузер')

    args = parser.parse_args()

    setup_paths(args.chapter, args.audio)

    print(f"\n{'='*50}")
    print(f"  Web Viewer Flask v{VERSION}")
    print(f"{'='*50}")
    print(f"  Глава: {CHAPTER_ID}")
    print(f"  JSON: {JSON_FILE}")
    print(f"  Аудио: {AUDIO_FILE or 'не указано'}")
    print(f"  URL: http://localhost:{args.port}")
    print(f"{'='*50}\n")

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    from werkzeug.serving import run_simple
    run_simple('localhost', args.port, app, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
