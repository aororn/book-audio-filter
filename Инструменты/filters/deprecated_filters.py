"""
Deprecated Filters v1.0 — Архив отключённых фильтров.

Здесь хранятся фильтры, которые были отключены из-за плохой точности.
Каждый фильтр документирован с причиной отключения и статистикой.

Этот файл НЕ импортируется в рабочий код — только для справки и истории.

История:
- v8.5.1 (2026-01-29): Отключено 6 агрессивных фильтров
- v6.1 (2026-01-28): Отключены фильтры grammar_ending, same_lemma, levenshtein
"""

VERSION = '1.0.0'
VERSION_DATE = '2026-01-30'


# =============================================================================
# ОТКЛЮЧЁННЫЕ ФИЛЬТРЫ v8.5.1 (2026-01-29)
# Причина: Плохое соотношение Golden / FP
# =============================================================================

class DeprecatedFilters:
    """
    Архив отключённых фильтров.

    НЕ ИСПОЛЬЗОВАТЬ в продакшене!
    Эти фильтры отключены из-за плохой точности.
    """

    # -------------------------------------------------------------------------
    # compound_prefix_insertion
    # Статистика: удаляет 51 golden и 17 FP
    # Соотношение: 3:1 (плохое, должно быть минимум 1:10)
    # -------------------------------------------------------------------------
    COMPOUND_PREFIXES_DEPRECATED = {
        'не', 'ни', 'по', 'на', 'за', 'от', 'до', 'пре', 'при',
    }

    @staticmethod
    def check_compound_prefix_insertion_DEPRECATED(error: dict) -> tuple:
        """
        ОТКЛЮЧЕНО v8.5.1: compound_prefix удаляет 51 golden и 17 FP

        Проверяет INS на составные приставки.
        Пример: вставленное "не" в "не знаю" может быть частью приставки.

        Причина отключения:
            Слишком много реальных ошибок чтеца (51 golden) фильтруются.
            Чтец действительно может вставить лишнее "не", "ни" и т.д.
        """
        return False, None

    # -------------------------------------------------------------------------
    # yandex_split_pairs
    # Статистика: 50% accuracy (11 golden, 11 FP)
    # Соотношение: 1:1 (очень плохое)
    # -------------------------------------------------------------------------
    YANDEX_SPLIT_PAIRS_DEPRECATED = {
        ('это', 'говори'): 'выторговали',
        ('вот', 'он'): 'жетон',
        ('там', 'и'): 'тамий',
    }

    @staticmethod
    def check_yandex_split_pairs_DEPRECATED(error: dict) -> tuple:
        """
        ОТКЛЮЧЕНО v8.5.1: 50% accuracy (11 golden, 11 FP)

        Проверяет INS из разбитых пар слов.
        Пример: "жетон" → "вот он", "выторговали" → "это говори"

        Причина отключения:
            Точность всего 50% — хуже случайного угадывания.
            Нельзя различить ошибку Яндекса от ошибки чтеца.
        """
        return False, None

    # -------------------------------------------------------------------------
    # duplicate_word_insertion
    # Статистика: удаляет 51 golden и только 3 FP
    # Соотношение: 17:1 (полностью вреден!)
    # -------------------------------------------------------------------------
    @staticmethod
    def check_duplicate_word_insertion_DEPRECATED(error: dict) -> tuple:
        """
        ОТКЛЮЧЕНО v8.5.1: удаляет 51 golden и только 3 FP

        Проверяет INS дублирующиеся слова (где где, там там).

        Причина отключения:
            ПОЛНОСТЬЮ ВРЕДЕН! Удаляет в 17 раз больше реальных ошибок чем FP.
            Дублирование слова часто является реальной ошибкой чтеца.
        """
        return False, None

    # -------------------------------------------------------------------------
    # yandex_particle_deletion
    # Статистика: удаляет 20 golden, 0 FP
    # Соотношение: ∞ (бесконечно вреден!)
    # -------------------------------------------------------------------------
    PARTICLES_DEPRECATED = {'же', 'ли', 'бы'}

    @staticmethod
    def check_yandex_particle_deletion_DEPRECATED(error: dict) -> tuple:
        """
        ОТКЛЮЧЕНО v8.5.1: удаляет 20 golden, 0 FP

        Проверяет DEL частиц "же"/"ли"/"бы" в середине предложения.
        Яндекс часто пропускает эти частицы: "Я же просунул" → "Я просунул"

        Причина отключения:
            ПОЛНОСТЬЮ ВРЕДЕН! Все срабатывания — реальные ошибки чтеца.
            Чтец часто пропускает частицы — это реальные ошибки.
        """
        return False, None

    # -------------------------------------------------------------------------
    # yandex_conjunction_before_gerund
    # Статистика: удаляет 51 golden, 0 FP
    # Соотношение: ∞ (бесконечно вреден!)
    # -------------------------------------------------------------------------
    @staticmethod
    def check_conjunction_before_gerund_DEPRECATED(error: dict) -> tuple:
        """
        ОТКЛЮЧЕНО v8.5.1: удаляет 51 golden, 0 FP

        Проверяет INS "и"/"а" перед деепричастием.
        Яндекс вставляет союз: "короткими фразами ведя" → "короткими фразами и ведя"

        Причина отключения:
            ПОЛНОСТЬЮ ВРЕДЕН! Все срабатывания — реальные ошибки чтеца.
            Чтец часто вставляет лишние союзы — это реальные ошибки.
        """
        return False, None


# =============================================================================
# ОТКЛЮЧЁННЫЕ ФИЛЬТРЫ v6.1 (2026-01-28)
# Причина: Заменены на smart_ версии в morpho_rules.py
# =============================================================================

class DeprecatedSmartFilters:
    """
    Архив фильтров, заменённых на smart_ версии.

    Эти фильтры работали, но были заменены на более точные версии
    в morpho_rules.py с проверкой грамматических различий.
    """

    @staticmethod
    def check_grammar_ending_match_DEPRECATED(w1: str, w2: str) -> tuple:
        """
        ОТКЛЮЧЕНО v6.1: заменено на smart_grammar в morpho_rules.py

        Проверяет совпадение по грамматическим окончаниям.

        Причина замены:
            Новая версия проверяет не только окончание, но и
            грамматические признаки (число, падеж, время).
            Разные грамматические формы = реальная ошибка чтеца.
        """
        return False, None

    @staticmethod
    def check_same_lemma_DEPRECATED(w1: str, w2: str) -> tuple:
        """
        ОТКЛЮЧЕНО v6.1: заменено на smart_lemma в morpho_rules.py

        Проверяет совпадение лемм.

        Причина замены:
            ОДИНАКОВАЯ ЛЕММА НЕ ОЗНАЧАЕТ ЛОЖНУЮ ОШИБКУ!
            Если лемма одинаковая, но есть грамматические различия —
            это РЕАЛЬНАЯ ошибка чтеца:
            - "сотни" → "сотня" — разное число
            - "теряю" → "теряя" — VERB → GRND
        """
        return False, None

    @staticmethod
    def check_similar_by_levenshtein_DEPRECATED(w1: str, w2: str) -> tuple:
        """
        ОТКЛЮЧЕНО v6.1: заменено на smart_levenshtein

        Проверяет схожесть по Левенштейну.

        Причина замены:
            Схожие слова могут быть реальными ошибками чтеца.
            Новая версия учитывает морфологию и контекст.
        """
        return False, None

    @staticmethod
    def check_prefix_variant_DEPRECATED(w1: str, w2: str) -> tuple:
        """
        ОТКЛЮЧЕНО v6.1: заменено на smart_prefix

        Проверяет варианты с приставками.

        Причина замены:
            Разные приставки меняют смысл слова.
            Новая версия проверяет семантику.
        """
        return False, None


# =============================================================================
# СТАТИСТИКА ОТКЛЮЧЁННЫХ ФИЛЬТРОВ
# =============================================================================

DEPRECATED_STATS = {
    'compound_prefix_insertion': {
        'version_disabled': 'v8.5.1',
        'golden_removed': 51,
        'fp_removed': 17,
        'ratio': '3:1 (плохое)',
        'status': 'PERMANENTLY_DISABLED',
    },
    'yandex_split_pairs': {
        'version_disabled': 'v8.5.1',
        'golden_removed': 11,
        'fp_removed': 11,
        'ratio': '1:1 (очень плохое)',
        'status': 'PERMANENTLY_DISABLED',
    },
    'duplicate_word_insertion': {
        'version_disabled': 'v8.5.1',
        'golden_removed': 51,
        'fp_removed': 3,
        'ratio': '17:1 (полностью вреден)',
        'status': 'PERMANENTLY_DISABLED',
    },
    'yandex_particle_deletion': {
        'version_disabled': 'v8.5.1',
        'golden_removed': 20,
        'fp_removed': 0,
        'ratio': '∞ (бесконечно вреден)',
        'status': 'PERMANENTLY_DISABLED',
    },
    'yandex_conjunction_before_gerund': {
        'version_disabled': 'v8.5.1',
        'golden_removed': 51,
        'fp_removed': 0,
        'ratio': '∞ (бесконечно вреден)',
        'status': 'PERMANENTLY_DISABLED',
    },
    'grammar_ending_match': {
        'version_disabled': 'v6.1',
        'replacement': 'morpho_rules.py::smart_grammar',
        'status': 'REPLACED',
    },
    'same_lemma': {
        'version_disabled': 'v6.1',
        'replacement': 'morpho_rules.py::smart_lemma',
        'status': 'REPLACED',
    },
    'similar_by_levenshtein': {
        'version_disabled': 'v6.1',
        'replacement': 'morpho_rules.py::smart_levenshtein',
        'status': 'REPLACED',
    },
    'prefix_variant': {
        'version_disabled': 'v6.1',
        'replacement': 'morpho_rules.py::smart_prefix',
        'status': 'REPLACED',
    },
}


def print_stats():
    """Выводит статистику отключённых фильтров."""
    print("=" * 60)
    print("СТАТИСТИКА ОТКЛЮЧЁННЫХ ФИЛЬТРОВ")
    print("=" * 60)

    total_golden_saved = 0
    total_fp_lost = 0

    for name, stats in DEPRECATED_STATS.items():
        print(f"\n{name}:")
        print(f"  Версия отключения: {stats.get('version_disabled', '?')}")

        if 'golden_removed' in stats:
            print(f"  Golden (реальных ошибок) удалялось: {stats['golden_removed']}")
            total_golden_saved += stats['golden_removed']

        if 'fp_removed' in stats:
            print(f"  FP (ложных ошибок) удалялось: {stats['fp_removed']}")
            total_fp_lost += stats['fp_removed']

        if 'ratio' in stats:
            print(f"  Соотношение: {stats['ratio']}")

        if 'replacement' in stats:
            print(f"  Замена: {stats['replacement']}")

        print(f"  Статус: {stats['status']}")

    print("\n" + "=" * 60)
    print(f"ИТОГО:")
    print(f"  Golden сохранено (отключением фильтров): {total_golden_saved}")
    print(f"  FP потеряно (отключением фильтров): {total_fp_lost}")
    print(f"  Чистая выгода: +{total_golden_saved} golden, -{total_fp_lost} FP")
    print("=" * 60)


if __name__ == '__main__':
    print_stats()
