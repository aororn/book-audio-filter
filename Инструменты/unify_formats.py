"""
Унификация форматов ошибок в проекте Яндекс Спич v1.0.

Стандартный формат ошибки:
- transcript: что услышал Яндекс (было 'wrong')
- original: что в книге (было 'correct')
- type: substitution | insertion | deletion | transposition
- time: время в секундах (float)
- time_seconds: время в секундах (int) — для совместимости
- context: контекст из оригинала

Алиасы для обратной совместимости:
- wrong = transcript
- correct = original

v1.0 (2026-01-31): Начальная версия
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

VERSION = '1.0.0'

# Маппинг старых ключей на новые (стандартные)
KEY_MAPPING = {
    'wrong': 'transcript',
    'correct': 'original',
}

# Обратный маппинг для совместимости
REVERSE_MAPPING = {v: k for k, v in KEY_MAPPING.items()}


def normalize_error(error: Dict[str, Any], add_aliases: bool = True) -> Dict[str, Any]:
    """
    Нормализует ошибку к стандартному формату.

    Args:
        error: Словарь с ошибкой (может иметь wrong/correct или transcript/original)
        add_aliases: Добавлять ли алиасы для обратной совместимости

    Returns:
        Нормализованный словарь
    """
    result = dict(error)

    # Конвертируем старые ключи в новые
    for old_key, new_key in KEY_MAPPING.items():
        if old_key in result:
            # Если нового ключа нет — копируем
            if new_key not in result:
                result[new_key] = result[old_key]
            # Удаляем старый ключ (если не нужны алиасы)
            if not add_aliases:
                del result[old_key]

    # Добавляем алиасы для обратной совместимости
    if add_aliases:
        for new_key, old_key in REVERSE_MAPPING.items():
            if new_key in result and old_key not in result:
                result[old_key] = result[new_key]

    # Нормализуем время
    if 'time' in result and 'time_seconds' not in result:
        result['time_seconds'] = int(result['time']) if isinstance(result['time'], (int, float)) else 0

    return result


def normalize_error_list(errors: List[Dict[str, Any]], add_aliases: bool = True) -> List[Dict[str, Any]]:
    """Нормализует список ошибок."""
    return [normalize_error(e, add_aliases) for e in errors]


def get_transcript(error: Dict[str, Any]) -> str:
    """Получает transcript (что услышал Яндекс) из любого формата."""
    return error.get('transcript', error.get('wrong', '')) or ''


def get_original(error: Dict[str, Any]) -> str:
    """Получает original (что в книге) из любого формата."""
    return error.get('original', error.get('correct', '')) or ''


def convert_golden_file(input_path: str, output_path: Optional[str] = None,
                        backup: bool = True) -> Dict[str, Any]:
    """
    Конвертирует golden файл в унифицированный формат.

    Args:
        input_path: Путь к golden файлу
        output_path: Путь для сохранения (если None — перезаписывает input)
        backup: Создавать ли бэкап

    Returns:
        Конвертированные данные
    """
    input_path = Path(input_path)

    # Читаем
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Конвертируем ошибки
    if 'errors' in data:
        original_errors = data['errors']
        data['errors'] = normalize_error_list(original_errors, add_aliases=True)

        # Добавляем метаданные о конверсии
        data['_format_version'] = '2.0'
        data['_normalized'] = True

    # Сохраняем бэкап
    if backup:
        backup_path = input_path.with_suffix('.json.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(input_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())

    # Сохраняем
    output_path = Path(output_path) if output_path else input_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def convert_all_golden_files(tests_dir: str = 'Тесты') -> int:
    """
    Конвертирует все golden файлы в директории.

    Returns:
        Количество конвертированных файлов
    """
    tests_path = Path(tests_dir)
    converted = 0

    for golden_file in tests_path.glob('золотой_стандарт_глава*.json'):
        print(f'Конвертирую: {golden_file.name}')
        try:
            convert_golden_file(str(golden_file))
            converted += 1
            print(f'  ✓ OK')
        except Exception as e:
            print(f'  ✗ Ошибка: {e}')

    return converted


def verify_format(file_path: str) -> Dict[str, Any]:
    """
    Проверяет формат файла и выводит статистику.

    Returns:
        Словарь со статистикой
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    errors = data.get('errors', data if isinstance(data, list) else [])

    stats = {
        'total_errors': len(errors),
        'has_transcript': 0,
        'has_original': 0,
        'has_wrong': 0,
        'has_correct': 0,
        'has_time': 0,
        'has_time_seconds': 0,
        'has_context': 0,
        'unified': 0,  # Ошибки с обоими форматами
    }

    for e in errors:
        if 'transcript' in e: stats['has_transcript'] += 1
        if 'original' in e: stats['has_original'] += 1
        if 'wrong' in e: stats['has_wrong'] += 1
        if 'correct' in e: stats['has_correct'] += 1
        if 'time' in e: stats['has_time'] += 1
        if 'time_seconds' in e: stats['has_time_seconds'] += 1
        if 'context' in e: stats['has_context'] += 1

        # Проверяем унифицированность
        if ('transcript' in e or 'wrong' in e) and ('original' in e or 'correct' in e):
            stats['unified'] += 1

    return stats


def print_format_report(file_path: str):
    """Выводит отчёт о формате файла."""
    stats = verify_format(file_path)

    print(f'\nФормат файла: {file_path}')
    print(f'  Всего ошибок: {stats["total_errors"]}')
    print(f'  С transcript: {stats["has_transcript"]}')
    print(f'  С original: {stats["has_original"]}')
    print(f'  С wrong: {stats["has_wrong"]}')
    print(f'  С correct: {stats["has_correct"]}')
    print(f'  С time: {stats["has_time"]}')
    print(f'  С time_seconds: {stats["has_time_seconds"]}')
    print(f'  Унифицированы: {stats["unified"]}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Унификация форматов ошибок')
    parser.add_argument('--convert-golden', action='store_true',
                        help='Конвертировать все golden файлы')
    parser.add_argument('--verify', type=str,
                        help='Проверить формат файла')
    parser.add_argument('--convert', type=str,
                        help='Конвертировать один файл')

    args = parser.parse_args()

    if args.convert_golden:
        print(f'Унификация форматов v{VERSION}')
        print('=' * 50)
        converted = convert_all_golden_files()
        print(f'\nКонвертировано файлов: {converted}')

    elif args.verify:
        print_format_report(args.verify)

    elif args.convert:
        convert_golden_file(args.convert)
        print(f'Конвертирован: {args.convert}')

    else:
        parser.print_help()
