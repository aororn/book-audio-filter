#!/usr/bin/env python3
"""
Audio Converter v2.0 - Конвертация аудио для Яндекс SpeechKit

Конвертирует аудиофайлы в оптимальный формат для распознавания:
- OggOpus (рекомендуется) — компактный, не требует указания параметров
- MP3 моно 64 kbps — альтернатива
- LPCM — для максимального качества

Использование:
    python audio_converter.py аудио.mp3 --format ogg
    python audio_converter.py аудио.mp3 --format mp3 --bitrate 64
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Импорт централизованной конфигурации
try:
    from config import (
        FileNaming, CHAPTERS_DIR,
        format_duration as config_format_duration,
        get_audio_info as config_get_audio_info,
        check_file_exists
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


# Рекомендуемые параметры для Яндекс SpeechKit
YANDEX_FORMATS = {
    'ogg': {
        'extension': '.ogg',
        'codec': 'libopus',
        'description': 'OggOpus (рекомендуется Яндексом)',
        'ffmpeg_args': ['-c:a', 'libopus', '-b:a', '48k', '-ac', '1', '-ar', '48000'],
    },
    'mp3': {
        'extension': '.mp3',
        'codec': 'libmp3lame',
        'description': 'MP3 моно',
        'ffmpeg_args': ['-c:a', 'libmp3lame', '-b:a', '64k', '-ac', '1', '-ar', '48000'],
    },
    'lpcm': {
        'extension': '.raw',
        'codec': 'pcm_s16le',
        'description': 'LPCM 16-bit (без заголовка)',
        'ffmpeg_args': ['-f', 's16le', '-c:a', 'pcm_s16le', '-ac', '1', '-ar', '48000'],
    },
    'wav': {
        'extension': '.wav',
        'codec': 'pcm_s16le',
        'description': 'WAV 16-bit моно',
        'ffmpeg_args': ['-c:a', 'pcm_s16le', '-ac', '1', '-ar', '48000'],
    },
}


def check_ffmpeg():
    """Проверяет наличие ffmpeg"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_audio_info(audio_path):
    """Получает информацию об аудиофайле через ffprobe."""
    # Используем функцию из config.py если доступна
    if HAS_CONFIG:
        info = config_get_audio_info(audio_path)
        # Добавляем bitrate и codec которые нужны для этого модуля
        if info.get('duration', -1) >= 0:
            try:
                import json
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format', '-show_streams',
                    str(audio_path)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    format_info = data.get('format', {})
                    info['bitrate'] = int(format_info.get('bit_rate', 0)) / 1000

                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'audio':
                            info['codec'] = stream.get('codec_name', '')
                            break
            except Exception:
                info['bitrate'] = 0
                info['codec'] = ''
        return info

    # Fallback логика
    try:
        import json
        result = subprocess.run([
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(audio_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            data = json.loads(result.stdout)

            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break

            format_info = data.get('format', {})

            return {
                'duration': float(format_info.get('duration', 0)),
                'size_mb': int(format_info.get('size', 0)) / 1024 / 1024,
                'bitrate': int(format_info.get('bit_rate', 0)) / 1000,
                'channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
                'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                'codec': audio_stream.get('codec_name', '') if audio_stream else '',
            }
    except Exception as e:
        print(f"Ошибка получения информации: {e}")

    return None


def format_duration(seconds):
    """Форматирует длительность в чч:мм:сс."""
    if HAS_CONFIG:
        return config_format_duration(seconds)

    # Fallback
    if seconds < 0:
        return "неизвестно"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def convert_audio(input_path, output_path=None, format='ogg', bitrate=None,
                  sample_rate=48000, verbose=True, force=False):
    """
    Конвертирует аудиофайл в формат для Яндекс SpeechKit.

    Args:
        input_path: путь к исходному аудио
        output_path: путь для сохранения (автоматически если не указан)
        format: целевой формат (ogg, mp3, lpcm, wav)
        bitrate: битрейт в kbps (для mp3/ogg)
        sample_rate: частота дискретизации
        verbose: выводить информацию
        force: перезаписать существующий файл без предупреждения

    Returns:
        путь к конвертированному файлу
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg не найден. Установите его:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    # Получаем информацию о исходном файле
    if verbose:
        info = get_audio_info(str(input_path))
        if info:
            print(f"\nИсходный файл: {input_path.name}")
            print(f"  Длительность: {format_duration(info['duration'])}")
            print(f"  Размер: {info['size_mb']:.1f} MB")
            print(f"  Битрейт: {info['bitrate']:.0f} kbps")
            print(f"  Каналы: {info['channels']}")
            print(f"  Частота: {info['sample_rate']} Hz")
            print(f"  Кодек: {info['codec']}")

    # Определяем формат
    if format not in YANDEX_FORMATS:
        raise ValueError(f"Неизвестный формат: {format}. Доступны: {list(YANDEX_FORMATS.keys())}")

    fmt = YANDEX_FORMATS[format]

    # Определяем выходной путь
    if output_path:
        output_path = Path(output_path)
    else:
        # Используем FileNaming для правильного имени
        if HAS_CONFIG and format == 'ogg':
            chapter_id = FileNaming.get_chapter_id(input_path)
            output_name = FileNaming.build_filename(chapter_id, 'audio')
            output_path = input_path.parent / output_name
        else:
            output_path = input_path.with_stem(input_path.stem + '_yandex').with_suffix(fmt['extension'])

    # Проверяем существование файла
    if output_path.exists() and not force:
        if HAS_CONFIG:
            if not check_file_exists(output_path, action='ask'):
                if verbose:
                    print(f"  → Файл уже существует, пропускаем: {output_path.name}")
                return str(output_path)
        else:
            if verbose:
                print(f"  ⚠ Файл уже существует: {output_path.name}")

    # Формируем команду ffmpeg
    ffmpeg_args = fmt['ffmpeg_args'].copy()

    # Переопределяем битрейт если указан
    if bitrate and format in ('mp3', 'ogg'):
        for i, arg in enumerate(ffmpeg_args):
            if arg == '-b:a':
                ffmpeg_args[i + 1] = f'{bitrate}k'

    # Переопределяем частоту дискретизации
    if sample_rate:
        for i, arg in enumerate(ffmpeg_args):
            if arg == '-ar':
                ffmpeg_args[i + 1] = str(sample_rate)

    cmd = [
        'ffmpeg', '-y',  # перезаписывать без вопросов
        '-i', str(input_path),
        *ffmpeg_args,
        str(output_path)
    ]

    if verbose:
        print(f"\nКонвертация в {fmt['description']}...")
        print(f"  Команда: {' '.join(cmd)}")

    # Выполняем конвертацию
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ошибка ffmpeg:\n{result.stderr}")

    # Проверяем результат
    if verbose and output_path.exists():
        out_info = get_audio_info(str(output_path))
        if out_info:
            print(f"\nРезультат: {output_path.name}")
            print(f"  Размер: {out_info['size_mb']:.1f} MB")
            print(f"  Битрейт: {out_info['bitrate']:.0f} kbps")
            print(f"  Каналы: {out_info['channels']} (моно)")
            print(f"  Частота: {out_info['sample_rate']} Hz")

            # Сжатие
            original_size = info['size_mb'] if info else 0
            if original_size > 0:
                compression = (1 - out_info['size_mb'] / original_size) * 100
                print(f"  Сжатие: {compression:.1f}%")

    return str(output_path)


def batch_convert(input_dir, output_dir=None, format='ogg', force=False, **kwargs):
    """Конвертирует все аудиофайлы в папке."""
    input_dir = Path(input_dir)

    # По умолчанию пишем рядом с исходниками, а не в подпапку
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = input_dir  # Рядом с исходниками

    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.wma'}
    # Исключаем .ogg если конвертируем в ogg (чтобы не конвертировать уже готовые)
    if format != 'ogg':
        audio_extensions.add('.ogg')

    audio_files = [f for f in input_dir.iterdir()
                   if f.suffix.lower() in audio_extensions and '_yandex' not in f.stem]

    if not audio_files:
        print(f"Аудиофайлы не найдены в {input_dir}")
        return []

    print(f"Найдено {len(audio_files)} аудиофайлов")

    results = []
    for audio_file in audio_files:
        # Используем FileNaming если доступен
        if HAS_CONFIG and format == 'ogg':
            chapter_id = FileNaming.get_chapter_id(audio_file)
            output_name = FileNaming.build_filename(chapter_id, 'audio')
            output_path = output_dir / output_name
        else:
            output_path = output_dir / (audio_file.stem + '_yandex' + YANDEX_FORMATS[format]['extension'])

        try:
            result = convert_audio(str(audio_file), str(output_path), format, force=force, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Ошибка конвертации {audio_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Конвертация аудио для Яндекс SpeechKit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Форматы:
  ogg   - OggOpus (рекомендуется) — компактный, отличное качество
  mp3   - MP3 моно — универсальный
  lpcm  - LPCM 16-bit — максимальное качество, большой размер
  wav   - WAV 16-bit — как LPCM, но с заголовком

Примеры:
  python audio_converter.py глава.mp3
  python audio_converter.py глава.mp3 --format ogg --bitrate 48
  python audio_converter.py --batch Главы/ --format ogg
        """
    )
    parser.add_argument('input', nargs='?', help='Входной аудиофайл или папка')
    parser.add_argument('--output', '-o', help='Выходной файл или папка')
    parser.add_argument('--format', '-f', default='ogg',
                        choices=list(YANDEX_FORMATS.keys()),
                        help='Формат (по умолчанию: ogg)')
    parser.add_argument('--bitrate', '-b', type=int, default=None,
                        help='Битрейт в kbps (для mp3/ogg)')
    parser.add_argument('--sample-rate', '-r', type=int, default=48000,
                        help='Частота дискретизации (по умолчанию: 48000)')
    parser.add_argument('--batch', action='store_true',
                        help='Обработать все файлы в папке')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Минимальный вывод')
    parser.add_argument('--force', action='store_true',
                        help='Перезаписать существующие файлы')

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        print("\n" + "="*50)
        print("Поддерживаемые форматы Яндекс SpeechKit:")
        for name, fmt in YANDEX_FORMATS.items():
            print(f"  {name:6} — {fmt['description']}")
        return

    try:
        if args.batch or Path(args.input).is_dir():
            batch_convert(
                args.input,
                args.output,
                format=args.format,
                bitrate=args.bitrate,
                sample_rate=args.sample_rate,
                verbose=not args.quiet,
                force=args.force
            )
        else:
            convert_audio(
                args.input,
                args.output,
                format=args.format,
                bitrate=args.bitrate,
                sample_rate=args.sample_rate,
                verbose=not args.quiet,
                force=args.force
            )

        print("\n✓ Готово!")

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
