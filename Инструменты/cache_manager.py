#!/usr/bin/env python3
"""
Cache Manager v2.0 - Модуль кэширования результатов

Кэширует результаты транскрибации и проверки для:
- Избежания повторной транскрибации одинаковых файлов
- Ускорения повторных проверок
- Сохранения истории изменений

Использование:
    from cache_manager import CacheManager
    cache = CacheManager()
    cache.set('key', data)
    data = cache.get('key')

Changelog:
    v2.0 (2026-01-24): Интеграция с config.py
        - TEMP_DIR для директории кэша
        - Замена md5 на sha256
        - Исправлены bare except clauses
        - Добавлен флаг --force в CLI
        - Добавлен вывод версии
    v1.0: Базовая реализация кэширования
"""

# Версия модуля
VERSION = '5.0.0'
VERSION_DATE = '2026-01-25'

import json
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta


# =============================================================================
# ИМПОРТ ЦЕНТРАЛИЗОВАННОЙ КОНФИГУРАЦИИ
# =============================================================================

try:
    from config import TEMP_DIR
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Fallback: папка рядом с модулем
    TEMP_DIR = Path(__file__).parent.parent / 'Темп'


class CacheManager:
    """Менеджер кэша для результатов обработки"""

    def __init__(self, cache_dir=None, max_age_days=30):
        """
        Инициализация менеджера кэша.

        Args:
            cache_dir: директория для кэша (по умолчанию TEMP_DIR/cache)
            max_age_days: максимальный возраст кэша в днях
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Используем TEMP_DIR из config.py или fallback
            self.cache_dir = TEMP_DIR / 'cache'

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        self.index_file = self.cache_dir / 'index.json'
        self._load_index()

    def _load_index(self):
        """Загружает индекс кэша"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except (json.JSONDecodeError, OSError, IOError) as e:
                print(f"  ⚠ Ошибка загрузки индекса кэша: {e}")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Сохраняет индекс кэша"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path):
        """Вычисляет хэш файла для идентификации (sha256)"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Читаем первые и последние 1MB для скорости
            hasher.update(f.read(1024 * 1024))
            f.seek(-min(1024 * 1024, os.path.getsize(file_path)), 2)
            hasher.update(f.read())

        # Добавляем размер файла
        hasher.update(str(os.path.getsize(file_path)).encode())

        return hasher.hexdigest()

    def _get_cache_path(self, key):
        """Возвращает путь к файлу кэша для ключа"""
        # Создаём безопасное имя файла (sha256)
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / f'{safe_key}.json'

    def has(self, key):
        """Проверяет, есть ли ключ в кэше"""
        if key not in self.index:
            return False

        entry = self.index[key]
        cache_path = self._get_cache_path(key)

        # Проверяем существование файла
        if not cache_path.exists():
            del self.index[key]
            self._save_index()
            return False

        # Проверяем возраст
        created = datetime.fromisoformat(entry.get('created', '2000-01-01'))
        if datetime.now() - created > self.max_age:
            self.delete(key)
            return False

        return True

    def get(self, key, default=None):
        """Получает данные из кэша"""
        if not self.has(key):
            return default

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Обновляем время доступа
            self.index[key]['accessed'] = datetime.now().isoformat()
            self._save_index()

            return data.get('data')
        except (json.JSONDecodeError, OSError, IOError, KeyError) as e:
            print(f"  ⚠ Ошибка чтения кэша для '{key}': {e}")
            return default

    def set(self, key, data, metadata=None):
        """Сохраняет данные в кэш"""
        cache_path = self._get_cache_path(key)

        cache_data = {
            'key': key,
            'data': data,
            'metadata': metadata or {},
            'created': datetime.now().isoformat(),
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        # Обновляем индекс
        self.index[key] = {
            'created': cache_data['created'],
            'accessed': cache_data['created'],
            'metadata': metadata or {},
        }
        self._save_index()

    def delete(self, key):
        """Удаляет ключ из кэша"""
        if key in self.index:
            del self.index[key]
            self._save_index()

        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
        except OSError as e:
            print(f"  ⚠ Ошибка удаления файла кэша: {e}")

    def clear(self):
        """Очищает весь кэш"""
        for key in list(self.index.keys()):
            self.delete(key)

    def cleanup(self):
        """Удаляет устаревшие записи кэша"""
        now = datetime.now()
        removed = 0

        for key in list(self.index.keys()):
            entry = self.index[key]
            created = datetime.fromisoformat(entry.get('created', '2000-01-01'))

            if now - created > self.max_age:
                self.delete(key)
                removed += 1

        return removed

    def get_for_file(self, file_path, cache_type='transcription'):
        """
        Получает кэшированные данные для файла.

        Args:
            file_path: путь к файлу
            cache_type: тип кэша (transcription, check, etc.)
        """
        file_hash = self._get_file_hash(file_path)
        key = f'{cache_type}:{file_hash}'
        return self.get(key)

    def set_for_file(self, file_path, data, cache_type='transcription'):
        """
        Сохраняет данные в кэш для файла.

        Args:
            file_path: путь к файлу
            data: данные для кэширования
            cache_type: тип кэша
        """
        file_hash = self._get_file_hash(file_path)
        key = f'{cache_type}:{file_hash}'

        metadata = {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'cache_type': cache_type,
        }

        self.set(key, data, metadata)

    def list_cached(self, cache_type=None):
        """Возвращает список кэшированных записей"""
        result = []

        for key, entry in self.index.items():
            if cache_type:
                if entry.get('metadata', {}).get('cache_type') != cache_type:
                    continue

            result.append({
                'key': key,
                **entry
            })

        return result

    def stats(self):
        """Возвращает статистику кэша"""
        total_size = 0
        for cache_file in self.cache_dir.glob('*.json'):
            if cache_file.name != 'index.json':
                total_size += cache_file.stat().st_size

        by_type = {}
        for key, entry in self.index.items():
            cache_type = entry.get('metadata', {}).get('cache_type', 'unknown')
            by_type[cache_type] = by_type.get(cache_type, 0) + 1

        return {
            'total_entries': len(self.index),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'by_type': by_type,
            'cache_dir': str(self.cache_dir),
        }


# Глобальный экземпляр кэша
_cache = None


def get_cache():
    """Возвращает глобальный экземпляр кэша"""
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def main():
    """CLI для управления кэшем"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Управление кэшем результатов обработки',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python cache_manager.py stats           # Статистика кэша
  python cache_manager.py list            # Список записей
  python cache_manager.py list -t transcription  # Фильтр по типу
  python cache_manager.py cleanup         # Удалить устаревшие
  python cache_manager.py clear --force   # Очистить весь кэш
        """
    )
    parser.add_argument('command', nargs='?', choices=['stats', 'list', 'cleanup', 'clear'],
                        help='Команда: stats, list, cleanup, clear')
    parser.add_argument('--type', '-t', help='Тип кэша для фильтрации')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Принудительное выполнение без подтверждения')
    parser.add_argument('--version', '-V', action='store_true',
                        help='Показать версию и выйти')

    args = parser.parse_args()

    # Вывод версии
    if args.version:
        print(f"Cache Manager v{VERSION} ({VERSION_DATE})")
        print(f"  Config: {'config.py' if HAS_CONFIG else 'fallback'}")
        return

    # Проверяем наличие команды
    if not args.command:
        parser.print_help()
        return

    cache = CacheManager()

    print(f"\n{'='*50}")
    print(f"  Cache Manager v{VERSION}")
    print(f"{'='*50}")

    if args.command == 'stats':
        stats = cache.stats()
        print(f"\nСтатистика кэша:")
        print(f"  Записей: {stats['total_entries']}")
        print(f"  Размер: {stats['total_size_mb']} MB")
        print(f"  Папка: {stats['cache_dir']}")
        print(f"\n  По типам:")
        for t, count in stats['by_type'].items():
            print(f"    {t}: {count}")

    elif args.command == 'list':
        entries = cache.list_cached(args.type)
        print(f"\nКэшированные записи ({len(entries)}):")
        for entry in entries[:20]:
            meta = entry.get('metadata', {})
            print(f"  - {meta.get('file_name', entry['key'][:30])}")
            print(f"    Тип: {meta.get('cache_type', '?')}, Создан: {entry.get('created', '?')}")

    elif args.command == 'cleanup':
        removed = cache.cleanup()
        print(f"Удалено устаревших записей: {removed}")

    elif args.command == 'clear':
        if args.force:
            cache.clear()
            print("Кэш очищен")
        else:
            confirm = input("Удалить весь кэш? (yes/no): ")
            if confirm.lower() == 'yes':
                cache.clear()
                print("Кэш очищен")
            else:
                print("Отменено")


if __name__ == '__main__':
    main()
