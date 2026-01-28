#!/usr/bin/env python3
"""
Инструмент для анализа ложных срабатываний фильтра.

Загружает отфильтрованные ошибки, извлекает контексты из всех источников,
классифицирует паттерны и проверяет конфликты с золотым стандартом.

Использование:
    python analyze_false_positives.py 01
    python analyze_false_positives.py 02 --limit 20
    python analyze_false_positives.py 01 --output report.md
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Добавляем путь к модулям фильтра
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from filters.constants import HOMOPHONES, YANDEX_TYPICAL_ERRORS


# =============================================================================
# КОНФИГУРАЦИЯ ПУТЕЙ
# =============================================================================

BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "Результаты проверки"
TESTS_DIR = BASE_DIR / "Тесты"


# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def find_chapter_files(chapter_id: str) -> Dict[str, Optional[Path]]:
    """Находит все файлы для указанной главы."""
    chapter_dir = RESULTS_DIR / chapter_id

    files = {
        'filtered': None,
        'transcript': None,
        'normalized': None,
        'golden': None,
    }

    # Filtered JSON
    filtered_path = chapter_dir / f"{chapter_id}_filtered.json"
    if filtered_path.exists():
        files['filtered'] = filtered_path

    # Transcript JSON
    transcript_path = chapter_dir / f"{chapter_id}_transcript.json"
    if transcript_path.exists():
        files['transcript'] = transcript_path

    # Normalized TXT (разные варианты имён)
    for pattern in [f"Глава {chapter_id}_normalized.txt",
                    f"Глава{chapter_id}_normalized.txt",
                    f"{chapter_id}_normalized.txt"]:
        norm_path = chapter_dir / pattern
        if norm_path.exists():
            files['normalized'] = norm_path
            break

    # Ищем normalized по glob
    if not files['normalized']:
        for f in chapter_dir.glob("*normalized*.txt"):
            files['normalized'] = f
            break

    # Golden standard (разные варианты имён)
    for pattern in [f"золотой_стандарт_глава{chapter_id}.json",
                    f"золотой_стандарт_глава{chapter_id.lstrip('0')}.json",
                    f"golden_{chapter_id}.json"]:
        golden_path = TESTS_DIR / pattern
        if golden_path.exists():
            files['golden'] = golden_path
            break

    return files


def load_filtered_errors(path: Path) -> List[Dict]:
    """Загружает отфильтрованные ошибки."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('errors', [])


def load_transcript(path: Path) -> List[Dict]:
    """Загружает транскрипцию Yandex."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words = []
    for chunk in data.get('chunks', []):
        for alt in chunk.get('alternatives', []):
            for word_info in alt.get('words', []):
                start_time = word_info.get('startTime', '0s')
                # Парсим время (формат "123.456s")
                if isinstance(start_time, str):
                    start_time = float(start_time.rstrip('s'))
                words.append({
                    'word': word_info.get('word', ''),
                    'time': start_time,
                    'confidence': word_info.get('confidence', 1.0),
                })
    return words


def load_normalized(path: Path) -> str:
    """Загружает нормализованный текст."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_golden_standard(path: Path) -> List[Dict]:
    """Загружает золотой стандарт."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('errors', [])


# =============================================================================
# ИЗВЛЕЧЕНИЕ КОНТЕКСТА
# =============================================================================

def extract_transcript_context(error: Dict, transcript_words: List[Dict],
                                window: int = 5) -> str:
    """Извлекает контекст из транскрипции по времени."""
    error_time = error.get('time', 0)

    # Находим ближайшие слова по времени
    nearby_words = []
    for word_info in transcript_words:
        word_time = word_info.get('time', 0)
        if abs(word_time - error_time) < 10:  # ±10 секунд
            nearby_words.append(word_info)

    if not nearby_words:
        return ""

    # Сортируем по времени
    nearby_words.sort(key=lambda x: x.get('time', 0))

    # Находим индекс ближайшего слова
    closest_idx = 0
    min_diff = float('inf')
    for i, word_info in enumerate(nearby_words):
        diff = abs(word_info.get('time', 0) - error_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    # Извлекаем окно
    start_idx = max(0, closest_idx - window)
    end_idx = min(len(nearby_words), closest_idx + window + 1)

    context_words = [w['word'] for w in nearby_words[start_idx:end_idx]]

    # Отмечаем центральное слово
    marker_pos = closest_idx - start_idx
    if 0 <= marker_pos < len(context_words):
        context_words[marker_pos] = f">>>{context_words[marker_pos]}<<<"

    return ' '.join(context_words)


def extract_normalized_context(error: Dict, normalized_text: str,
                                window: int = 50) -> str:
    """Извлекает контекст из нормализованного текста."""
    # Используем original из ошибки
    original = error.get('original', '')
    if not original:
        return ""

    # Ищем в нормализованном тексте (без регистра)
    text_lower = normalized_text.lower()
    original_lower = original.lower()

    pos = text_lower.find(original_lower)
    if pos == -1:
        return f"[НЕ НАЙДЕНО: {original}]"

    # Извлекаем контекст
    start = max(0, pos - window)
    end = min(len(normalized_text), pos + len(original) + window)

    context = normalized_text[start:end]

    # Отмечаем найденное слово
    word_start = pos - start
    word_end = word_start + len(original)
    context = (context[:word_start] + ">>>" +
               context[word_start:word_end] + "<<<" +
               context[word_end:])

    return context.replace('\n', ' ')


def extract_original_context(error: Dict) -> str:
    """Извлекает контекст из поля context ошибки."""
    context = error.get('context', '')
    original = error.get('original', '')

    if not context:
        return ""

    # Отмечаем слово в контексте
    if original and original in context:
        context = context.replace(original, f">>>{original}<<<", 1)

    return context


# =============================================================================
# КЛАССИФИКАЦИЯ ПАТТЕРНОВ
# =============================================================================

def classify_error_pattern(error: Dict) -> Tuple[str, str]:
    """
    Классифицирует ошибку по типу паттерна.

    Возвращает: (pattern_type, recommendation)
    """
    error_type = error.get('type', '')
    original = error.get('original', '').lower()
    transcript = error.get('transcript', '').lower()

    # 1. Проверяем существующие омофоны
    for pair in HOMOPHONES:
        if (original == pair[0] and transcript == pair[1]) or \
           (original == pair[1] and transcript == pair[0]):
            return "ОМОФОН (уже в словаре)", "Уже отфильтровано - проверить фильтр"

    # 2. Проверяем типичные ошибки Яндекса
    for pair in YANDEX_TYPICAL_ERRORS:
        if (original == pair[0] and transcript == pair[1]) or \
           (original == pair[1] and transcript == pair[0]):
            return "YANDEX_ERROR (уже в словаре)", "Уже отфильтровано - проверить фильтр"

    # 3. Анализируем тип замены
    if error_type == 'substitution':
        # Проверяем окончания
        if len(original) > 2 and len(transcript) > 2:
            if original[:-1] == transcript[:-1]:
                return "ОКОНЧАНИЕ", f"Возможно грамматическое окончание"
            if original[:-2] == transcript[:-2]:
                return "ОКОНЧАНИЕ-2", f"Возможно падежная форма"

        # Проверяем префиксы
        if original.startswith(transcript) or transcript.startswith(original):
            return "ПРЕФИКС", f"Добавить в YANDEX_TYPICAL_ERRORS: ('{original}', '{transcript}')"

        # Проверяем фонетическое сходство (грубая оценка)
        common_chars = set(original) & set(transcript)
        similarity = len(common_chars) / max(len(set(original)), len(set(transcript)))
        if similarity > 0.7:
            return "ФОНЕТИКА", f"Возможно добавить в HOMOPHONES: ('{original}', '{transcript}')"

        # Местоимения
        pronouns = {'он', 'она', 'оно', 'они', 'я', 'ты', 'мы', 'вы'}
        if original in pronouns or transcript in pronouns:
            return "МЕСТОИМЕНИЕ", f"Добавить в HOMOPHONES: ('{original}', '{transcript}')"

        return "НЕИЗВЕСТНО", "Требует ручного анализа"

    elif error_type == 'insertion':
        word = transcript
        # Слабые слова
        weak_words = {'то', 'же', 'ли', 'бы', 'вот', 'вон', 'ну', 'и', 'а', 'но'}
        if word in weak_words:
            return "ВСТАВКА-СЛАБОЕ", "Возможно уже фильтруется WEAK_INSERTIONS"
        return "ВСТАВКА", "Требует ручного анализа"

    elif error_type == 'deletion':
        word = original
        return "УДАЛЕНИЕ", "Требует ручного анализа"

    return "НЕИЗВЕСТНО", "Требует ручного анализа"


# =============================================================================
# ПРОВЕРКА КОНФЛИКТОВ
# =============================================================================

def check_golden_conflict(error: Dict, golden_errors: List[Dict]) -> bool:
    """
    Проверяет, есть ли конфликт с золотым стандартом.

    Возвращает True если добавление этого паттерна может
    отфильтровать реальную ошибку из золотого стандарта.
    """
    original = error.get('original', '').lower()
    transcript = error.get('transcript', '').lower()
    error_type = error.get('type', '')

    for golden_error in golden_errors:
        # Золотой стандарт использует wrong/correct вместо transcript/original
        g_original = (golden_error.get('correct', '') or
                      golden_error.get('original', '')).lower()
        g_transcript = (golden_error.get('wrong', '') or
                        golden_error.get('transcript', '')).lower()
        g_type = golden_error.get('type', '')

        # Прямое совпадение
        if g_type == error_type:
            if g_original == original and g_transcript == transcript:
                return True
            # Обратное совпадение (паттерн симметричный)
            if g_original == transcript and g_transcript == original:
                return True

    return False


# =============================================================================
# ФОРМАТИРОВАНИЕ ОТЧЁТА
# =============================================================================

def format_error_card(error: Dict, index: int,
                       transcript_ctx: str, normalized_ctx: str, original_ctx: str,
                       pattern_type: str, recommendation: str,
                       has_conflict: bool) -> str:
    """Форматирует карточку ошибки."""
    time = error.get('time', 0)
    error_type = error.get('type', '')
    original = error.get('original', '')
    transcript = error.get('transcript', '')

    lines = []
    lines.append("┌" + "─" * 78 + "┐")
    lines.append(f"│  ЛОЖНОЕ СРАБАТЫВАНИЕ #{index:<3} — {time:.2f}с" + " " * (78 - 35 - len(f"{time:.2f}")) + "│")
    lines.append(f"│  Тип: {error_type:<20}" + " " * (78 - 28) + "│")

    if error_type == 'substitution':
        change = f"{transcript} → {original}"
        lines.append(f"│  Найдено: {change:<67}│")
    elif error_type == 'insertion':
        lines.append(f"│  Вставлено: {transcript:<65}│")
    elif error_type == 'deletion':
        lines.append(f"│  Удалено: {original:<67}│")

    lines.append("│" + " " * 78 + "│")

    # Контексты
    if original_ctx:
        ctx_line = f"│  ОРИГИНАЛ:    {original_ctx[:60]}"
        lines.append(ctx_line + " " * (79 - len(ctx_line)) + "│")

    if normalized_ctx:
        ctx_line = f"│  НОРМ.ТЕКСТ:  {normalized_ctx[:60]}"
        lines.append(ctx_line + " " * (79 - len(ctx_line)) + "│")

    if transcript_ctx:
        ctx_line = f"│  ТРАНСКРИПТ:  {transcript_ctx[:60]}"
        lines.append(ctx_line + " " * (79 - len(ctx_line)) + "│")

    lines.append("│" + " " * 78 + "│")

    # Паттерн и рекомендация
    pattern_line = f"│  Паттерн: {pattern_type:<67}│"
    lines.append(pattern_line)

    conflict_str = "ДА ⚠️" if has_conflict else "НЕТ ✓"
    conflict_line = f"│  Конфликт с золотым: {conflict_str:<55}│"
    lines.append(conflict_line)

    rec_line = f"│  Рекомендация: {recommendation[:61]}"
    lines.append(rec_line + " " * (79 - len(rec_line)) + "│")

    lines.append("└" + "─" * 78 + "┘")

    return "\n".join(lines)


def generate_summary(errors: List[Dict], patterns: Dict[str, int],
                     conflicts: int) -> str:
    """Генерирует сводку по анализу."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("                           СВОДКА АНАЛИЗА")
    lines.append("=" * 80)
    lines.append(f"\nВсего проанализировано ошибок: {len(errors)}")
    lines.append(f"Конфликтов с золотым стандартом: {conflicts}")
    lines.append("\nРаспределение по паттернам:")

    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        lines.append(f"  {pattern}: {count}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def analyze_chapter(chapter_id: str, limit: Optional[int] = None,
                    output_path: Optional[Path] = None,
                    exclude_golden: bool = False) -> str:
    """Анализирует ложные срабатывания для главы.

    Args:
        chapter_id: Идентификатор главы (01, 02, ...)
        limit: Ограничение количества ошибок
        output_path: Путь для сохранения отчёта
        exclude_golden: Исключить ошибки, совпадающие с золотым стандартом
    """

    # Находим файлы
    files = find_chapter_files(chapter_id)

    if not files['filtered']:
        return f"Ошибка: не найден файл {chapter_id}_filtered.json"

    # Загружаем данные
    errors = load_filtered_errors(files['filtered'])

    transcript_words = []
    if files['transcript']:
        transcript_words = load_transcript(files['transcript'])

    normalized_text = ""
    if files['normalized']:
        normalized_text = load_normalized(files['normalized'])

    golden_errors = []
    if files['golden']:
        golden_errors = load_golden_standard(files['golden'])

    # Ограничиваем количество
    if limit:
        errors = errors[:limit]

    # Анализируем каждую ошибку
    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"  АНАЛИЗ ЛОЖНЫХ СРАБАТЫВАНИЙ — ГЛАВА {chapter_id}")
    report_lines.append(f"  Файлов найдено: filtered={bool(files['filtered'])}, "
                       f"transcript={bool(files['transcript'])}, "
                       f"normalized={bool(files['normalized'])}, "
                       f"golden={bool(files['golden'])}")
    report_lines.append(f"{'='*80}\n")

    patterns_count: Dict[str, int] = {}
    conflicts_count = 0

    displayed_count = 0
    for i, error in enumerate(errors, 1):
        # Проверяем конфликт заранее для фильтрации
        has_conflict = check_golden_conflict(error, golden_errors)

        # Пропускаем ошибки из золотого стандарта если запрошено
        if exclude_golden and has_conflict:
            conflicts_count += 1
            continue

        if has_conflict:
            conflicts_count += 1

        # Извлекаем контексты
        transcript_ctx = extract_transcript_context(error, transcript_words)
        normalized_ctx = extract_normalized_context(error, normalized_text)
        original_ctx = extract_original_context(error)

        # Классифицируем
        pattern_type, recommendation = classify_error_pattern(error)
        patterns_count[pattern_type] = patterns_count.get(pattern_type, 0) + 1

        displayed_count += 1

        # Форматируем карточку
        card = format_error_card(
            error, displayed_count,
            transcript_ctx, normalized_ctx, original_ctx,
            pattern_type, recommendation,
            has_conflict
        )
        report_lines.append(card)
        report_lines.append("")

    # Добавляем сводку
    summary = generate_summary(errors, patterns_count, conflicts_count)
    report_lines.append(summary)

    report = "\n".join(report_lines)

    # Сохраняем если указан путь
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчёт сохранён: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Анализ ложных срабатываний фильтра"
    )
    parser.add_argument(
        "chapter",
        help="ID главы (например: 01, 02)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Ограничить количество анализируемых ошибок"
    )
    parser.add_argument(
        "--no-golden",
        action="store_true",
        help="Исключить ошибки из золотого стандарта (показать только настоящие ложные срабатывания)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь для сохранения отчёта"
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    exclude_golden = getattr(args, 'no_golden', False)
    report = analyze_chapter(args.chapter, args.limit, output_path, exclude_golden)
    print(report)


if __name__ == "__main__":
    main()
