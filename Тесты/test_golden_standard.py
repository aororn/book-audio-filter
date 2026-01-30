#!/usr/bin/env python3
"""
Тест золотого стандарта v6.5

Проверяет, что система находит ВСЕ ошибки из эталонного списка.
Если какая-то ошибка не найдена — тест провален.

Использование:
    python test_golden_standard.py отчет.json золотой_стандарт.json
    python test_golden_standard.py отчет.json  # использует стандарт из папки Тесты

v6.5 (2026-01-31):
- Использует унифицированный формат ошибок из unify_formats.py
- Поддержка как transcript/original, так и wrong/correct
- Добавлены функции get_transcript(), get_original()
"""

import argparse
import json
import sys
from pathlib import Path

# Импортируем унифицированные функции
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'Инструменты'))
    from unify_formats import get_transcript, get_original
except ImportError:
    # Fallback если unify_formats недоступен
    def get_transcript(error):
        return error.get('transcript', error.get('wrong', '')) or ''
    def get_original(error):
        return error.get('original', error.get('correct', '')) or ''


def normalize(word):
    """Нормализует слово для сравнения"""
    return word.lower().strip().replace('ё', 'е')


def time_to_seconds(time_str):
    """Преобразует время 'M:SS' в секунды"""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(time_str)


def words_match(expected_word, found_word):
    """
    Проверяет совпадение слов с учётом многословных полей.

    Для коротких слов (<=2 символа) требует точное совпадение,
    чтобы избежать ложных совпадений вроде "а" in "правило".

    Для длинных слов допускает вхождение (для случаев "рагидон" / "рагедон").
    """
    if not expected_word or not found_word:
        return False

    # Точное совпадение
    if expected_word == found_word:
        return True

    # Для коротких слов — только точное совпадение или пословное
    if len(expected_word) <= 2 or len(found_word) <= 2:
        # Проверяем пословно: "а" должно совпадать со словом "а",
        # но не с "правило" (которое содержит букву "а")
        exp_words = set(expected_word.split())
        found_words = set(found_word.split())
        return bool(exp_words & found_words)

    # Для длинных слов — допускаем вхождение подстроки
    return expected_word in found_word or found_word in expected_word


def find_matching_error(expected, found_errors, time_tolerance=10):
    """
    Ищет ошибку из золотого стандарта среди найденных.

    Args:
        expected: ожидаемая ошибка из стандарта
        found_errors: список найденных ошибок
        time_tolerance: допустимое отклонение по времени (секунды)

    Returns:
        найденная ошибка или None
    """
    exp_time = expected.get('time_seconds', time_to_seconds(expected.get('time', 0)))
    # v6.5: Используем унифицированные функции для получения полей
    exp_wrong = normalize(get_transcript(expected))
    exp_correct = normalize(get_original(expected))
    exp_type = expected.get('type', 'substitution')

    for err in found_errors:
        err_time = err.get('time', 0)
        err_type = err.get('type', '')

        # Проверяем время (с допуском)
        if abs(err_time - exp_time) > time_tolerance:
            continue

        # v6.5: Используем унифицированные функции для получения полей
        # Поддержка обоих форматов: wrong/correct и transcript/original
        err_wrong = normalize(get_transcript(err))
        err_correct = normalize(get_original(err))

        # Прямое совпадение по типу
        if err_type == exp_type:
            if exp_type == 'substitution':
                # Совпадение по неправильному слову
                if words_match(exp_wrong, err_wrong):
                    return err
                # Совпадение по правильному слову
                if words_match(exp_correct, err_correct):
                    return err

            elif exp_type == 'deletion':
                # deletion: wrong='', correct=пропущенное слово
                if words_match(exp_correct, err_correct):
                    return err

            elif exp_type == 'insertion':
                # insertion: wrong=лишнее слово, correct=''
                if words_match(exp_wrong, err_wrong):
                    return err

            elif exp_type == 'transposition':
                # transposition: перестановка слов (wrong и correct содержат те же слова в разном порядке)
                # Проверяем, что оба набора слов совпадают
                if exp_wrong and exp_correct and err_wrong and err_correct:
                    exp_words = set(exp_wrong.split())
                    err_words = set(err_wrong.split()) | set(err_correct.split())
                    if exp_words & err_words:  # пересечение не пусто
                        return err

        # Особый случай: substitution может быть найдена как insertion+deletion
        # Например: ТЫ→что (substitution) может быть найдена как:
        #   - deletion →что (пропущено "что")
        #   - insertion ты→ (лишнее "ты")
        if exp_type == 'substitution':
            if err_type == 'deletion':
                # deletion нашла correct из substitution
                if words_match(exp_correct, err_correct):
                    return err
            elif err_type == 'insertion':
                # insertion нашла wrong из substitution
                if words_match(exp_wrong, err_wrong):
                    return err
            elif err_type == 'transposition':
                # transposition может содержать слова из substitution
                # Пример: "и"→"я" найдено как "я и"→"и я"
                err_all_words = set(err_wrong.split()) | set(err_correct.split())
                if exp_wrong in err_all_words or exp_correct in err_all_words:
                    return err

        # v6.4: Обратный случай: insertion или deletion могут быть найдены как substitution
        # Это происходит когда smart_compare.merge_adjacent_ins_del объединяет
        # соседние insertion + deletion в одну substitution.
        # Например: golden имеет deletion "→затем" + insertion "он→"
        #           а система нашла substitution "он→затем"
        if exp_type == 'deletion' and err_type == 'substitution':
            # deletion: correct = пропущенное слово
            # substitution: original = что в книге (= пропущенное слово)
            if words_match(exp_correct, err_correct):
                return err
        if exp_type == 'insertion' and err_type == 'substitution':
            # insertion: wrong = лишнее слово
            # substitution: transcript = что услышал Яндекс (= лишнее слово)
            if words_match(exp_wrong, err_wrong):
                return err

    return None


def test_golden_standard(report_path, standard_path, verbose=True):
    """
    Тестирует отчёт на соответствие золотому стандарту.

    Returns:
        (passed: bool, found: int, total: int, missing: list)
    """
    # Загружаем отчёт
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Загружаем золотой стандарт
    with open(standard_path, 'r', encoding='utf-8') as f:
        standard = json.load(f)

    found_errors = report.get('errors', [])
    expected_errors = standard.get('errors', [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ТЕСТ ЗОЛОТОГО СТАНДАРТА")
        print(f"{'='*60}")
        print(f"  Отчёт: {report_path}")
        print(f"  Стандарт: {standard_path}")
        print(f"  Ожидаемых ошибок: {len(expected_errors)}")
        print(f"  Найденных ошибок: {len(found_errors)}")
        print(f"{'='*60}\n")

    # Проверяем каждую ожидаемую ошибку
    found_count = 0
    missing = []

    for exp in expected_errors:
        match = find_matching_error(exp, found_errors)

        # v6.5: Используем унифицированные функции
        exp_transcript = get_transcript(exp)
        exp_original = get_original(exp)

        if match:
            found_count += 1
            if verbose:
                print(f"  ✓ {exp['time']} — {exp_transcript} → {exp_original}")
        else:
            missing.append(exp)
            if verbose:
                print(f"  ✗ {exp['time']} — {exp_transcript} → {exp_original} — НЕ НАЙДЕНА!")

    # Результат
    passed = len(missing) == 0
    total = len(expected_errors)

    if verbose:
        print(f"\n{'='*60}")
        if passed:
            print(f"  ✓ ТЕСТ ПРОЙДЕН: {found_count}/{total} ошибок найдено")
        else:
            print(f"  ✗ ТЕСТ НЕ ПРОЙДЕН: {found_count}/{total} ошибок найдено")
            print(f"\n  Пропущенные ошибки:")
            for err in missing:
                # v6.5: Используем унифицированные функции
                print(f"    - {err['time']}: {get_transcript(err)} → {get_original(err)}")
                if err.get('context'):
                    print(f"      Контекст: {err['context'][:60]}...")
        print(f"{'='*60}\n")

    return passed, found_count, total, missing


def main():
    parser = argparse.ArgumentParser(
        description='Тест золотого стандарта ошибок'
    )
    parser.add_argument('report', help='JSON файл с отчётом проверки')
    parser.add_argument('standard', nargs='?', help='JSON файл с золотым стандартом')
    parser.add_argument('--tolerance', '-t', type=int, default=10,
                        help='Допуск по времени в секундах (по умолчанию: 10)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Минимальный вывод')

    args = parser.parse_args()

    # Ищем золотой стандарт
    if args.standard:
        standard_path = args.standard
    else:
        # Пробуем найти в папке Тесты
        project_dir = Path(__file__).parent.parent
        standard_path = project_dir / 'Тесты' / 'золотой_стандарт_глава1.json'
        if not standard_path.exists():
            print("Золотой стандарт не найден. Укажите путь явно.")
            sys.exit(1)

    passed, found, total, missing = test_golden_standard(
        args.report,
        str(standard_path),
        verbose=not args.quiet
    )

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
