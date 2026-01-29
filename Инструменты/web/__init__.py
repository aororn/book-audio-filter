"""
Web пакет v1.0 - Модульная структура веб-интерфейса

Разделение web_viewer.py на модули:
- context_enricher.py — обогащение контекстов ошибок из оригинала
- html_generator.py — генерация HTML страницы
- server.py — HTTP сервер и обработчики запросов

Основные экспорты:
- enrich_errors_with_original — обогащение контекстов
- generate_html — генерация HTML
- run_server — запуск веб-сервера

Changelog:
    v1.0 (2026-01-25): Начальная версия, разделение web_viewer.py
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-25'

from .context_enricher import (
    enrich_errors_with_original,
    find_original_context,
    find_insertion_context_in_original,
    find_deletion_context_in_original,
    load_original_text,
    load_transcript_words,
    normalize_for_search,
    find_word_position_in_context,
)

from .html_generator import (
    generate_html,
    load_html_template,
    escape_error_data,
)

from .server import (
    run_server,
    CustomHandler,
    auto_detect_related_files,
)

__all__ = [
    # Context enricher
    'enrich_errors_with_original',
    'find_original_context',
    'find_insertion_context_in_original',
    'find_deletion_context_in_original',
    'load_original_text',
    'load_transcript_words',
    'normalize_for_search',
    'find_word_position_in_context',

    # HTML generator
    'generate_html',
    'load_html_template',
    'escape_error_data',

    # Server
    'run_server',
    'CustomHandler',
    'auto_detect_related_files',

    # Version
    'VERSION',
    'VERSION_DATE',
]
