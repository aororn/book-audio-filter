"""
Конфигурация фильтрации v1.3.

Централизованное хранение всех порогов, констант и настроек фильтрации.
Заменяет разбросанные по engine.py магические числа.

v1.3 (2026-01-31): Добавлены пороги из phonetic_semantic.py и safety_veto.py
v1.2 (2026-01-31): Добавлены пороги из window_verifier, sliding_window, frequency_manager, smart_scorer
v1.1 (2026-01-31): Заменены константы-алиасы на функции get_*_threshold()
v1.0 (2026-01-31): Начальная версия
"""

from dataclasses import dataclass, field
from typing import Set, Optional

VERSION = '1.3.0'


@dataclass
class FilterThresholds:
    """Пороги для фильтрации."""

    # ML-классификатор
    # v12.1: Эксперимент с 85% НЕ ПРОШЁЛ — "услышав → услышал" отфильтровано
    # Оставляем консервативный порог 90%
    ml_confidence: float = 0.90

    # Семантическая защита оговорок
    # v8.9: Калиброванные пороги на основе анализа БД (941 ошибок)
    # high semantic + diff_lemma = 12 golden, 247 FP
    semantic_slip: float = 0.4
    phonetic_slip: float = 0.7

    # Левенштейн
    levenshtein_default: int = 2
    levenshtein_name_ratio: float = 0.5  # max(3, int(max_len * 0.5))

    # Минимальные длины слов
    min_word_length_strict: int = 3
    min_word_length_name: int = 3
    min_word_length_suffix: int = 4
    min_word_length_fragment: int = 2

    # Контекстная верификация
    context_anchor_distance: int = 2  # Максимальное расстояние для якорей

    # Морфологическая когерентность
    morpho_agreement_diff_threshold: float = 0.3
    morpho_agreement_fp_threshold: float = -0.5
    morpho_agreement_min: float = 0.7

    # Семантическая связность
    semantic_coherence_diff: float = 0.15
    semantic_coherence_trans_min: float = 0.25
    semantic_coherence_orig_max: float = 0.35

    # Misrecognition artifact
    misrecognition_ratio: float = 0.6

    # WindowVerifier (v1.2)
    window_technical_ok: float = 0.95    # >= 95% схожесть = технический шум
    window_error: float = 0.70           # < 70% = явная ошибка

    # SlidingWindow (v1.2)
    phonetic_match: int = 95             # fuzz.ratio >= 95% = фонетическое совпадение
    substring_match: int = 90            # покрытие для подстрок

    # FrequencyManager (v1.2)
    freq_rare: int = 10                  # ipm — редкое слово
    freq_bookish: int = 50               # ipm — книжное слово

    # SmartScorer (v1.2)
    smart_filter_default: int = 60       # порог для SmartFilter

    # ScoringEngine (v1.2)
    scoring_filter: int = 50             # penalty >= 50 = не фильтровать
    scoring_high_confidence: float = 0.85
    scoring_medium_confidence: float = 0.75
    scoring_low_confidence: float = 0.65

    # SemanticManager (v1.2)
    synonym_threshold: float = 0.5       # порог синонимии

    # PhoneticSemantic (v1.3)
    # Пороги для фонетико-семантических фильтров (phonetic_semantic.py)
    phon_sem_high_phon: float = 0.8      # порог высокой фонетики
    phon_sem_high_sem: float = 0.5       # порог высокой семантики
    phon_sem_perfect_phon: float = 0.99  # порог идеальной фонетики

    # SafetyVeto (v1.3)
    # Пороги для финальной защиты (safety_veto.py)
    veto_semantic_slip: float = 0.4      # порог оговорки (semantic)
    veto_phonetic_exception: float = 0.8 # порог исключения из оговорок (phonetic)
    veto_semantic_exception: float = 0.5 # порог исключения из оговорок (semantic)


@dataclass
class FilterFlags:
    """Флаги включения/выключения модулей."""

    use_lemmatization: bool = True
    use_homophones: bool = True
    use_ml_classifier: bool = True
    use_context_verifier: bool = True
    use_semantic_manager: bool = True
    use_smart_filter: bool = False  # Отключён в v9.2
    use_morpho_coherence: bool = True
    use_semantic_coherence: bool = True
    use_phonetic_morphoform: bool = True


@dataclass
class VersionRequirements:
    """Минимальные версии зависимостей."""

    min_smart_compare: str = '10.5.0'
    min_context_verifier: str = '4.0.0'
    min_ml_classifier: str = '1.1.0'


@dataclass
class FilterConfig:
    """
    Главная конфигурация фильтрации.

    Использование:
        config = FilterConfig()
        config.thresholds.ml_confidence = 0.85
        config.flags.use_ml_classifier = False
    """

    thresholds: FilterThresholds = field(default_factory=FilterThresholds)
    flags: FilterFlags = field(default_factory=FilterFlags)
    versions: VersionRequirements = field(default_factory=VersionRequirements)

    # Дополнительные настройки из внешнего конфига
    protected_words: Optional[Set[str]] = None
    weak_words: Optional[Set[str]] = None

    def merge_dict(self, config_dict: dict) -> 'FilterConfig':
        """
        Мержит настройки из словаря (для совместимости с существующим API).

        Args:
            config_dict: словарь с настройками (levenshtein_threshold, use_lemmatization, etc.)

        Returns:
            self для цепочки вызовов
        """
        if 'levenshtein_threshold' in config_dict:
            self.thresholds.levenshtein_default = config_dict['levenshtein_threshold']

        if 'use_lemmatization' in config_dict:
            self.flags.use_lemmatization = config_dict['use_lemmatization']

        if 'use_homophones' in config_dict:
            self.flags.use_homophones = config_dict['use_homophones']

        if 'protected_words' in config_dict:
            self.protected_words = config_dict['protected_words']

        if 'weak_words' in config_dict:
            self.weak_words = config_dict['weak_words']

        return self

    def to_dict(self) -> dict:
        """
        Конвертирует в словарь (для совместимости с существующим API).
        """
        return {
            'levenshtein_threshold': self.thresholds.levenshtein_default,
            'use_lemmatization': self.flags.use_lemmatization,
            'use_homophones': self.flags.use_homophones,
            'protected_words': self.protected_words,
            'weak_words': self.weak_words,
        }


# Глобальный экземпляр по умолчанию
_default_config: Optional[FilterConfig] = None


def get_default_config() -> FilterConfig:
    """Возвращает глобальную конфигурацию по умолчанию."""
    global _default_config
    if _default_config is None:
        _default_config = FilterConfig()
    return _default_config


def set_default_config(config: FilterConfig) -> None:
    """Устанавливает глобальную конфигурацию."""
    global _default_config
    _default_config = config


# Функции для доступа к порогам (вместо констант, которые не обновляются)
def get_ml_threshold() -> float:
    """Возвращает текущий порог ML-классификатора."""
    return get_default_config().thresholds.ml_confidence

def get_semantic_slip_threshold() -> float:
    """Возвращает текущий порог семантической защиты."""
    return get_default_config().thresholds.semantic_slip

def get_phonetic_slip_threshold() -> float:
    """Возвращает текущий порог фонетической защиты."""
    return get_default_config().thresholds.phonetic_slip

# v1.2: Дополнительные функции для доступа к порогам
def get_window_technical_ok_threshold() -> float:
    """WindowVerifier: порог технического шума."""
    return get_default_config().thresholds.window_technical_ok

def get_window_error_threshold() -> float:
    """WindowVerifier: порог явной ошибки."""
    return get_default_config().thresholds.window_error

def get_phonetic_match_threshold() -> int:
    """SlidingWindow: порог фонетического совпадения."""
    return get_default_config().thresholds.phonetic_match

def get_substring_match_threshold() -> int:
    """SlidingWindow: порог подстроки."""
    return get_default_config().thresholds.substring_match

def get_freq_rare_threshold() -> int:
    """FrequencyManager: порог редкого слова."""
    return get_default_config().thresholds.freq_rare

def get_freq_bookish_threshold() -> int:
    """FrequencyManager: порог книжного слова."""
    return get_default_config().thresholds.freq_bookish

def get_smart_filter_threshold() -> int:
    """SmartScorer: порог по умолчанию."""
    return get_default_config().thresholds.smart_filter_default

def get_scoring_filter_threshold() -> int:
    """ScoringEngine: порог фильтрации."""
    return get_default_config().thresholds.scoring_filter

def get_synonym_threshold() -> float:
    """SemanticManager: порог синонимии."""
    return get_default_config().thresholds.synonym_threshold

# v1.3: PhoneticSemantic пороги
def get_phon_sem_high_phon() -> float:
    """PhoneticSemantic: порог высокой фонетики."""
    return get_default_config().thresholds.phon_sem_high_phon

def get_phon_sem_high_sem() -> float:
    """PhoneticSemantic: порог высокой семантики."""
    return get_default_config().thresholds.phon_sem_high_sem

def get_phon_sem_perfect_phon() -> float:
    """PhoneticSemantic: порог идеальной фонетики."""
    return get_default_config().thresholds.phon_sem_perfect_phon

# v1.3: SafetyVeto пороги
def get_veto_semantic_slip() -> float:
    """SafetyVeto: порог оговорки."""
    return get_default_config().thresholds.veto_semantic_slip

def get_veto_phonetic_exception() -> float:
    """SafetyVeto: порог исключения (phonetic)."""
    return get_default_config().thresholds.veto_phonetic_exception

def get_veto_semantic_exception() -> float:
    """SafetyVeto: порог исключения (semantic)."""
    return get_default_config().thresholds.veto_semantic_exception


if __name__ == '__main__':
    # Тест
    config = FilterConfig()
    print(f"FilterConfig v{VERSION}")
    print(f"  ML threshold: {config.thresholds.ml_confidence}")
    print(f"  Semantic slip: {config.thresholds.semantic_slip}")
    print(f"  Use ML: {config.flags.use_ml_classifier}")

    # Тест merge
    config.merge_dict({'levenshtein_threshold': 3, 'use_lemmatization': False})
    print(f"  After merge:")
    print(f"    Levenshtein: {config.thresholds.levenshtein_default}")
    print(f"    Use lemma: {config.flags.use_lemmatization}")
