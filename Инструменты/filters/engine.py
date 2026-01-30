"""
Движок фильтрации ошибок транскрипции v9.3.2.

Содержит:
- should_filter_error — решение по одной ошибке (оркестратор правил)
- filter_errors — фильтрация списка ошибок
- filter_report — фильтрация JSON-отчёта

v9.3.2 изменения (2026-01-30):
- РЕФАКТОРИНГ: Убран костыль "как то ... там" (строка 404)
  - Обобщено: любое "{prefix} то там" теперь фильтруется как разбитое выражение
  - Было: только "как то там"
  - Стало: "что то там", "где то там", "когда то там" и т.д.
  - Документация добавлена в код

v9.2 изменения (2026-01-30):
- ИНТЕГРАЦИЯ: ML-классификатор (ml_classifier.py) как уровень 10
  - RandomForest на 22 признаках, порог 90%
  - Фильтрует только с высокой уверенностью
- ИНТЕГРАЦИЯ: SmartFilter (smart_filter.py) как уровень 11
  - Полноценная система на накопительном скоринге
  - Частотный анализ + семантика + скользящее окно

v9.1 изменения (2026-01-30):
- МИГРАЦИЯ: Inline-код заменён на вызовы rules/ модулей:
  - SAFE_ENDING_TRANSITIONS → rules.check_safe_ending_transition()
  - YANDEX_PHONETIC_PAIRS → rules.check_yandex_phonetic_pair()
  - alignment_artifact → rules.check_alignment_artifact()
  - single_consonant_artifact → rules.check_single_consonant_artifact()
  - i_ya_confusion → rules.check_i_ya_confusion()
- Убраны дублирующие inline-константы

v9.0 изменения (2026-01-30):
- РЕФАКТОРИНГ: Правила вынесены в модули rules/
  - rules/protection.py — защитные слои
  - rules/phonetics.py — фонетические пары
  - rules/alignment.py — артефакты выравнивания
- engine.py теперь ОРКЕСТРАТОР, а не монолит

v8.9 изменения (2026-01-30):
- Интеграция SemanticManager: семантическая близость как ЗАЩИТНЫЙ слой
  (высокая семантика + разные леммы = оговорка чтеца = НЕ фильтровать)
- Интеграция SmartScorer: комплексный скоринг для метрик
- БЕЗОПАСНО: не меняет фильтрацию, добавляет защиту golden

v8.8 изменения (2026-01-30):
- Добавлен фильтр misrecognition_artifact для артефактов распознавания
  (вставленное слово похоже на соседнее слово в контексте: "блядочное"~"ублюдочные")
- Использует SequenceMatcher для сравнения (порог 0.6)
- БЕЗОПАСНО: проверено на БД — 0 golden, 6 FP

v8.7 изменения (2026-01-30):
- Добавлен фильтр single_consonant_artifact для однобуквенных согласных
  (артефакты выравнивания типа -"с", -"м", -"в", -"п", -"к", -"ф", -"х", -"э")
- БЕЗОПАСНО: не затрагивает golden (проверено на БД)

v8.4 изменения:
- alignment_artifact_substring теперь проверяет леммы:
  если леммы равны (господином/господин) — это грамматика, НЕ артефакт

v8.3 изменения:
- Исключены частицы составных слов (то/нибудь/либо) из alignment_artifact

v8.2 изменения:
- Фонетические пары Яндекса: не/ни, ну/но, а/о и т.д.
- Артефакты выравнивания: короткое vs длинное, подстрока

v8.1 изменения:
- Интеграция ScoringEngine: HARD_NEGATIVES как защитный уровень
- Известные пары путаницы (сотни/сотня, получится/получилось) защищены от фильтрации

v8.0 изменения:
- Единый модуль morpho_rules.py вместо smart_rules + learned_rules
- Консервативная фильтрация: при любом грамматическом различии НЕ фильтруем
- Протестировано на 70 golden ошибках — ни одна не фильтруется

v7.0 изменения (устарело):
- Интеграция learned_rules.py — обученные правила на 614 парах данных

v6.0 изменения (устарело):
- Интеграция smart_rules.py для алгоритмических правил
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any, Set

from .constants import (
    PROTECTED_WORDS, WEAK_WORDS,
    ALIGNMENT_ARTIFACTS_DEL, SHORT_WEAK_WORDS, WEAK_CONJUNCTIONS,
    ALIGNMENT_ARTIFACTS_INS, WEAK_INSERTIONS, FUNCTION_WORDS,
    YANDEX_SPLIT_INSERTIONS, SENTENCE_START_WEAK_WORDS,
    YANDEX_SPLIT_PAIRS, INTERROGATIVE_PRONOUNS, RARE_ADVERBS,
    SKIP_SPLIT_FRAGMENT,  # v11.8: перемещено из локальной переменной
)
from .comparison import (
    normalize_word, get_word_info, get_lemma, get_pos, get_number, get_case,
    levenshtein_distance, parse_word_cached,
    is_homophone_match, is_grammar_ending_match, is_case_form_match,
    is_adverb_adjective_match, is_verb_gerund_safe_match,
    is_short_full_adjective_match, is_lemma_match,
    is_similar_by_levenshtein, is_yandex_typical_error,
    is_prefix_variant, is_interjection,
    HAS_PYMORPHY, morph,
)
from .detectors import (
    is_yandex_name_error, is_merged_word_error, is_compound_word_match,
    is_split_name_insertion, is_compound_prefix_insertion,
    is_split_compound_insertion, is_context_artifact,
    detect_alignment_chains, detect_linked_prefix_errors,
    FULL_CHARACTER_NAMES, CHARACTER_NAMES_BASE,
)
from .morpho_rules import get_morpho_rules, is_morpho_false_positive

# v9.0: Импорт модульных правил из rules/
try:
    from .rules import (
        apply_protection_layers,
        check_yandex_phonetic_pair,
        check_i_ya_confusion,
        check_alignment_artifact,
        check_safe_ending_transition,
        check_single_consonant_artifact,
        YANDEX_PHONETIC_PAIRS,
        SAFE_ENDING_TRANSITIONS,
        COMPOUND_PARTICLES,
        SINGLE_CONSONANT_ARTIFACTS,
    )
    HAS_RULES_MODULE = True
except ImportError:
    HAS_RULES_MODULE = False

# Версия модуля
VERSION = '9.3.2'
VERSION_DATE = '2026-01-30'

# v9.3.2 изменения:
# - РЕФАКТОРИНГ: Убран костыль "как то ... там"
#   - Обобщено: любое "{prefix} то там" теперь фильтруется
#   - Было: if prefix.startswith('как') and next_word == 'там'
#   - Стало: if next_word == 'там' (для любого prefix)

# v9.3.1 изменения:
# - ЗАЩИТА: substitution с merged_from_ins_del=True пропускают alignment_artifact
#   - Эти ошибки созданы merge_adjacent_ins_del (smart_compare v10.7.2)
#   - Пример: deletion "затем" + insertion "он" → substitution "он" → "затем"
#   - Раньше фильтровались alignment_artifact_length (разная длина слов)
#   - Теперь защищены — это реальные ошибки чтеца

# v9.3.0 изменения:
# - ИНТЕГРАЦИЯ: ContextVerifier v1.0 как уровень 12
#   - Контекстная верификация артефактов склеенных/разбитых слов
#   - Работает только для insertion
#   - Безопасно: 0 golden затронуто, 27+ FP отфильтровано на главе 5

# v9.2.2 изменения:
# - safe_ending_transition: добавлена проверка падежа (get_case)
# - check_alignment_artifact: добавлен get_case_func
# - ML защита: same_lemma + diff_number → не применяем ML

# v9.2.1 изменения:
# - Эксперимент ML 85% НЕ ПРОШЁЛ: "услышав → услышал" отфильтровано
# - Порог возвращён на 90%

# Минимальная совместимая версия smart_compare для валидации
MIN_SMART_COMPARE_VERSION = '10.5.0'

# v8.1: Импорт ScoringEngine для защиты имён и hard negatives
try:
    from .scoring_engine import (
        should_filter_by_score, is_hard_negative, HARD_NEGATIVES
    )
    HAS_SCORING_ENGINE = True
except ImportError:
    HAS_SCORING_ENGINE = False

# v8.9: Импорт SemanticManager для семантической защиты оговорок
try:
    from .semantic_manager import get_semantic_manager, get_similarity
    HAS_SEMANTIC_MANAGER = True
except ImportError:
    HAS_SEMANTIC_MANAGER = False
    get_similarity = lambda w1, w2: 0.0

# v8.9: Импорт SmartScorer для комплексного скоринга
try:
    from .smart_scorer import SmartScorer, ScoreResult, WEIGHTS as SCORER_WEIGHTS
    HAS_SMART_SCORER = True
except ImportError:
    HAS_SMART_SCORER = False

# v11.8: Импорт SmartFilter для полноценной фильтрации
try:
    from .smart_filter import SmartFilter, get_smart_filter, evaluate_error_smart
    HAS_SMART_FILTER = True
except ImportError:
    HAS_SMART_FILTER = False

# v12.1: Импорт ContextVerifier для контекстной верификации
try:
    from .context_verifier import should_filter_by_context, VERSION as CONTEXT_VERIFIER_VERSION
    HAS_CONTEXT_VERIFIER = True
except ImportError:
    HAS_CONTEXT_VERIFIER = False
    CONTEXT_VERIFIER_VERSION = 'N/A'

# v11.8: Импорт ML-классификатора
try:
    import sys
    from pathlib import Path
    _parent_dir = Path(__file__).parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from ml_classifier import get_classifier, FalsePositiveClassifier
    _ml_classifier = get_classifier()
    HAS_ML_CLASSIFIER = _ml_classifier.model is not None
except Exception:
    HAS_ML_CLASSIFIER = False
    _ml_classifier = None

# Порог уверенности для ML-классификатора
# v12.1: Эксперимент с 85% — НЕ ПРОШЁЛ (2026-01-30)
# При 85% отфильтровалась golden ошибка "услышав → услышал" (86.1%)
# Оставляем консервативный порог 90%
ML_CONFIDENCE_THRESHOLD = 0.90

# v8.9: Калиброванные пороги на основе анализа БД (941 ошибок)
# Анализ: high semantic + diff_lemma = 12 golden, 247 FP
# Безопасный порог: semantic >= 0.4 + phonetic >= 0.7 = оговорка
SEMANTIC_SLIP_THRESHOLD = 0.4      # Семантическая близость для оговорки
PHONETIC_SLIP_THRESHOLD = 0.7      # Фонетическая близость для оговорки


def should_filter_error(
    error: Dict[str, Any],
    config: Optional[Dict] = None,
    all_errors: Optional[List[Dict]] = None,
) -> Tuple[bool, str]:
    """Определяет, нужно ли отфильтровать ошибку."""
    config = config or {}
    levenshtein_threshold = config.get('levenshtein_threshold', 2)
    use_lemmatization = config.get('use_lemmatization', True)
    use_homophones = config.get('use_homophones', True)
    protected_words = config.get('protected_words', PROTECTED_WORDS)
    weak_words = config.get('weak_words', WEAK_WORDS)

    error_type = error.get('type', '')

    # Получаем слова
    if error_type == 'substitution':
        word1 = error.get('wrong', '') or error.get('transcript', '')
        word2 = error.get('correct', '') or error.get('original', '')
        words = [word1, word2]
    elif error_type == 'insertion':
        word = error.get('wrong', '') or error.get('transcript', '') or error.get('word', '')
        words = [word]
        word1, word2 = word, ''
    elif error_type == 'deletion':
        word = error.get('correct', '') or error.get('original', '') or error.get('word', '')
        words = [word]
        word1, word2 = '', word
    else:
        word = error.get('word', '')
        words = [word]
        word1 = word2 = word

    words_norm = [normalize_word(w) for w in words]

    # ==== УРОВЕНЬ -1: ScoringEngine ЗАЩИТА (v8.1) ====
    # Проверяем HARD_NEGATIVES — известные пары путаницы, которые нельзя фильтровать
    # Это защитный уровень: если пара в HARD_NEGATIVES — ПРЕКРАЩАЕМ фильтрацию
    if HAS_SCORING_ENGINE and error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]
        if is_hard_negative(w1, w2):
            # Это известная пара путаницы — НЕ фильтруем, это реальная ошибка
            return False, 'PROTECTED_hard_negative'

    # ==== УРОВЕНЬ -0.6: Междометия (v9.2.4) ====
    # Пары междометий (кхм↔хм, ах↔ох) — всегда фильтруем как техническую ошибку
    # Междометия не могут быть "оговорками" — они не несут семантического смысла
    if error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]
        if is_interjection(w1) and is_interjection(w2):
            return True, 'interjection_pair'

    # ==== УРОВЕНЬ -0.5: SemanticManager ЗАЩИТА (v8.9) ====
    # Высокая семантическая близость + разные леммы = оговорка чтеца
    # Анализ БД: semantic>0.4 + diff_lemma = реальные ошибки (мечтательны→мечтатели)
    # Это ЗАЩИТНЫЙ уровень: НЕ фильтруем оговорки
    if HAS_SEMANTIC_MANAGER and error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]
        # Только для разных лемм — проверяем семантику
        if HAS_PYMORPHY:
            lemma1 = get_lemma(w1)
            lemma2 = get_lemma(w2)
            if lemma1 and lemma2 and lemma1 != lemma2:
                semantic_sim = get_similarity(w1, w2)
                # Если семантика высокая — это оговорка, не фильтруем
                if semantic_sim >= SEMANTIC_SLIP_THRESHOLD:
                    return False, f'PROTECTED_semantic_slip({semantic_sim:.2f})'

    # ==== УРОВЕНЬ 0: Morpho Rules (v8.0) — консервативная фильтрация ====
    # Фильтруем ТОЛЬКО если 100% уверены в ложной ошибке
    # При любом грамматическом различии — НЕ фильтруем
    if error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]

        morpho_result = get_morpho_rules().check(w1, w2)
        if morpho_result and morpho_result.should_filter:
            return True, f'morpho_{morpho_result.rule_name}'

    # ==== УРОВЕНЬ 0.3: Безопасные окончания (v8.8 → v9.1 migrated to rules/) ====
    # Переходы окончаний, которые встречаются ТОЛЬКО в FP, НИКОГДА в Golden
    # v9.2.1: добавлена проверка падежа (get_case)
    if HAS_RULES_MODULE and error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]
        should_filter, reason = check_safe_ending_transition(
            w1, w2,
            get_lemma_func=get_lemma if HAS_PYMORPHY else None,
            get_pos_func=get_pos if HAS_PYMORPHY else None,
            get_case_func=get_case if HAS_PYMORPHY else None
        )
        if should_filter:
            return True, reason

    # ==== УРОВЕНЬ 0.5: Фонетические пары Яндекса (v8.2 → v9.1 migrated to rules/) ====
    if HAS_RULES_MODULE and error_type == 'substitution' and len(words_norm) >= 2:
        should_filter, reason = check_yandex_phonetic_pair(words_norm[0], words_norm[1])
        if should_filter:
            return True, reason

    # ==== УРОВЕНЬ 0.6: Артефакты выравнивания (v8.4 → v9.1 migrated to rules/) ====
    # v9.2.1: добавлена проверка падежа (get_case)
    # v9.3.2: защита merged_from_ins_del — если разные леммы, это реальная ошибка
    is_merged = error.get('merged_from_ins_del', False)
    if HAS_RULES_MODULE and error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]

        # Защита merged: разные леммы = реальная ошибка чтеца, не артефакт
        skip_alignment = False
        if is_merged and HAS_PYMORPHY:
            lemma1 = get_lemma(w1)
            lemma2 = get_lemma(w2)
            if lemma1 and lemma2 and lemma1 != lemma2:
                skip_alignment = True  # Разные леммы — не фильтруем

        if not skip_alignment:
            should_filter, reason = check_alignment_artifact(
                w1, w2,
                error_type='substitution',
                get_lemma_func=get_lemma if HAS_PYMORPHY else None,
                get_pos_func=get_pos if HAS_PYMORPHY else None,
                get_case_func=get_case if HAS_PYMORPHY else None
        )
        if should_filter:
            return True, reason

    # ==== ЭТАП 0: Артефакты алгоритма выравнивания ====

    error_time = error.get('time', 0)
    if error_type == 'deletion' and error_time == 0:
        return True, 'alignment_start_artifact'

    # DEL имён персонажей — проверяем и базовые, и полные формы (минимум 3 символа)
    if error_type == 'deletion' and len(words_norm[0]) >= 3:
        if words_norm[0] in CHARACTER_NAMES_BASE or words_norm[0] in FULL_CHARACTER_NAMES:
            return True, 'character_name_unrecognized'

    if error_type == 'insertion' and words_norm[0]:
        inserted_word = words_norm[0]
        if inserted_word not in FUNCTION_WORDS:
            transcript_ctx = error.get('transcript_context', '').lower()
            if transcript_ctx:
                ctx_words = transcript_ctx.split()
                for i, ctx_word in enumerate(ctx_words):
                    if ctx_word == inserted_word:
                        if i > 0:
                            prev_word = ctx_words[i - 1]
                            combined = prev_word + inserted_word
                            if combined in FULL_CHARACTER_NAMES:
                                return True, 'split_name_insertion'
                            for name in FULL_CHARACTER_NAMES:
                                if len(name) >= 6 and levenshtein_distance(combined, name) <= 1:
                                    return True, 'split_name_insertion'
                        if i < len(ctx_words) - 1:
                            next_word = ctx_words[i + 1]
                            combined = inserted_word + next_word
                            if combined in FULL_CHARACTER_NAMES:
                                return True, 'split_name_insertion'
                            for name in FULL_CHARACTER_NAMES:
                                if len(name) >= 6 and levenshtein_distance(combined, name) <= 1:
                                    return True, 'split_name_insertion'
                        break

    # INS "то" — compound_particle_to (расширенный)
    if error_type == 'insertion' and words_norm[0] == 'то':
        # Проверяем в transcript_context паттерн "кто то", "что то" и т.д.
        transcript_ctx = error.get('transcript_context', '').lower()
        original_ctx = error.get('context', '').lower()
        for pronoun in INTERROGATIVE_PRONOUNS:
            pattern = f'{pronoun} то'
            if pattern in transcript_ctx:
                # Проверяем, что в оригинале ЕСТЬ дефисное слово (кто-то)
                # Если есть — Яндекс разбил его, это ложная вставка
                # Если нет — чтец реально вставил "то", это настоящая ошибка
                hyphenated = f'{pronoun}-то'
                if hyphenated in original_ctx:
                    return True, 'interrogative_split_to'
        # Старая логика для compound_particle_to
        # v9.3.2: Рефакторинг костыля "как то ... там"
        # Устойчивые выражения: "как-то там", "что-то там", "где-то там" и т.д.
        # Яндекс разбивает их на "как то там" — это ложная вставка "то"
        context = error.get('context', '').lower()
        compound_prefixes = [
            'что', 'как', 'кто', 'где', 'когда', 'куда', 'откуда', 'почему', 'зачем',
            'какой', 'какая', 'какое', 'какие',
        ]
        for prefix in compound_prefixes:
            pattern = f'{prefix} то'
            if pattern in context:
                idx = context.find(pattern)
                after_to_start = idx + len(pattern)
                after_to = context[after_to_start:].strip().split()
                if after_to:
                    next_word = after_to[0]
                    # v9.3.2: Устойчивое выражение "{prefix}-то там" — фильтруем
                    # Это разбитое Яндексом "как-то там", "что-то там", "где-то там"
                    if next_word == 'там':
                        return True, 'compound_particle_to'
                    # Если после "то" идёт направление или глагол — это реальная вставка "то"
                    # Пример: "кто сунется то туда" — чтец реально вставил лишнее "то"
                    direction_words = {'туда', 'сюда', 'тут', 'здесь', 'теперь', 'тогда'}
                    verb_endings = ('ся', 'ет', 'ит', 'ут', 'ат', 'ют', 'ёт')
                    if next_word in direction_words or next_word.endswith(verb_endings):
                        continue
                return True, 'compound_particle_to'

    if error_type == 'insertion':
        transcript_context = error.get('transcript_context', '')
        if is_split_name_insertion(words_norm[0], transcript_context):
            return True, 'split_name'

    # ОТКЛЮЧЕНО v8.5.1: compound_prefix — см. deprecated_filters.py

    if error_type == 'insertion':
        transcript_context = error.get('transcript_context', '')
        original_context = error.get('context', '')
        if is_split_compound_insertion(words_norm[0], transcript_context, original_context):
            return True, 'split_compound'

    if error_type == 'insertion' and words_norm[0] in YANDEX_SPLIT_INSERTIONS:
        expected_prev = YANDEX_SPLIT_INSERTIONS[words_norm[0]]
        transcript_ctx = error.get('transcript_context', '').lower()
        pattern = f'{expected_prev} {words_norm[0]}'
        if pattern in transcript_ctx:
            return True, 'split_word_yandex'

    # ОТКЛЮЧЕНО v8.5.1: yandex_split_pairs — см. deprecated_filters.py

    # INS как суффикс разбитого слова (говори от выторговали)
    if error_type == 'insertion' and len(words_norm[0]) >= 4:
        inserted = words_norm[0]
        original_ctx = error.get('context', '').lower()
        # Ищем в оригинале слова, которые заканчиваются на вставленное
        ctx_words = original_ctx.split()
        for ctx_word in ctx_words:
            ctx_clean = normalize_word(ctx_word)
            if len(ctx_clean) >= len(inserted) + 3 and ctx_clean.endswith(inserted):
                return True, 'split_suffix_insertion'

    # ОТКЛЮЧЕНО v8.5.1: duplicate_word_insertion — см. deprecated_filters.py

    # v5.3: INS коротких слов от разбиения длинных (мы от "големы", ли от "или")
    # v5.4: НЕ применяем к однобуквенным союзам/частицам — это настоящие ошибки чтеца
    # v11.8: SKIP_SPLIT_FRAGMENT перемещено в constants.py
    if error_type == 'insertion' and len(words_norm[0]) == 2 and words_norm[0] not in SKIP_SPLIT_FRAGMENT:
        inserted = words_norm[0]
        transcript_ctx = error.get('transcript_context', '').lower()
        original_ctx = error.get('context', '').lower()
        # Ищем в транскрипте слово, оканчивающееся на inserted (голе мы = големы)
        trans_words = transcript_ctx.split()
        orig_words = original_ctx.split()
        for i, tw in enumerate(trans_words):
            if tw == inserted and i > 0:
                prev_trans = trans_words[i - 1]
                combined = prev_trans + inserted
                # Проверяем, есть ли ТОЧНОЕ совпадение (без Левенштейна — слишком много ложных)
                for ow in orig_words:
                    ow_clean = normalize_word(ow)
                    if ow_clean == combined:
                        return True, 'split_word_fragment'
                break

    # ==== ЭТАП 3: Междометия ====
    if error_type == 'deletion':
        if is_interjection(words_norm[0]):
            return True, 'interjection'

    # v8.7: Однобуквенные согласные — артефакты выравнивания (→ v9.1 migrated to rules/)
    if HAS_RULES_MODULE and error_type in ('deletion', 'insertion'):
        word_to_check = words_norm[0]
        should_filter, reason = check_single_consonant_artifact(word_to_check)
        if should_filter:
            return True, reason

    # v8.8: Артефакты распознавания — вставленное слово похоже на слово в контексте
    # Примеры: "блядочное"~"ублюдочные", "оголим"~"големах", "смертник"~"пересмешник"
    # Безопасно: проверено на БД — 0 golden, 8 FP
    if error_type == 'insertion' and len(words_norm[0]) >= 4:
        inserted = words_norm[0]
        context = error.get('context', '').lower()
        if context:
            ctx_words = context.split()
            for ctx_word in ctx_words:
                ctx_clean = ''.join(c for c in ctx_word if c.isalpha())
                # Пропускаем короткие слова и само вставленное слово
                if len(ctx_clean) >= 4 and ctx_clean != inserted:
                    ratio = SequenceMatcher(None, inserted, ctx_clean).ratio()
                    if ratio > 0.6:
                        return True, 'misrecognition_artifact'

    # v8.8: Вставка неизвестного слова (UNKN) — артефакт распознавания
    # Примеры: "бла" (обрыв "глава")
    # Безопасно: проверено на БД — 0 golden, 1 FP
    if error_type == 'insertion' and HAS_PYMORPHY and len(words_norm[0]) >= 2:
        inserted = words_norm[0]
        parsed = morph.parse(inserted)
        if parsed and 'UNKN' in str(parsed[0].tag):
            return True, 'unknown_word_artifact'

    # DEL редких наречий (эдак, этак)
    if error_type == 'deletion' and words_norm[0] in RARE_ADVERBS:
        return True, 'rare_adverb'

    # DEL слабых слов в начале нового предложения (после ./?/!)
    if error_type == 'deletion' and words_norm[0] in SENTENCE_START_WEAK_WORDS:
        context = error.get('context', '')
        marker_pos = error.get('marker_pos', -1)
        if marker_pos > 0:
            before_context = context[:marker_pos].rstrip()
            if before_context and before_context[-1] in '.!?':
                # Слово стоит в начале нового предложения после ./?/!
                return True, 'sentence_start_weak'

    # DEL частей дефисных слов (тесь от Займи-тесь)
    if error_type == 'deletion' and len(words_norm[0]) >= 2:
        deleted = words_norm[0]
        context = error.get('context', '')
        # Проверяем паттерны: "-слово" или "слово-"
        if f'-{deleted}' in context.lower() or f'{deleted}-' in context.lower():
            return True, 'hyphenated_part'

    # ОТКЛЮЧЕНО v8.5.1: yandex_particle_deletion — см. deprecated_filters.py

    # v5.3: DEL частей составных слов (возвышение от само+возвышение, звёздной от шести+звёздной)
    if error_type == 'deletion' and len(words_norm[0]) >= 4:
        deleted = words_norm[0]
        context = error.get('context', '').lower()
        transcript_ctx = error.get('transcript_context', '').lower()
        # Ищем в оригинале или транскрипте составное слово, содержащее deleted
        ctx_words = context.split() + transcript_ctx.split()
        for cw in ctx_words:
            cw_clean = normalize_word(cw)
            # Проверяем: cw заканчивается на deleted (само+возвышение)
            if len(cw_clean) > len(deleted) + 2 and cw_clean.endswith(deleted):
                # Убеждаемся, что это не просто deleted + что-то
                prefix = cw_clean[:-len(deleted)]
                if len(prefix) >= 2:
                    return True, 'compound_word_part'
            # Проверяем: cw начинается с deleted (звёздной+фракции)
            if len(cw_clean) > len(deleted) + 2 and cw_clean.startswith(deleted):
                suffix = cw_clean[len(deleted):]
                if len(suffix) >= 2:
                    return True, 'compound_word_part'

    # ==== ЭТАП 4: Контекстные фильтры ====
    if is_context_artifact(error, all_errors):
        return True, 'context_artifact'

    # ==== УРОВЕНЬ 1: Защищённые слова ====
    has_protected = any(w in protected_words for w in words_norm)

    if error_type == 'substitution' and has_protected:
        w1, w2 = words_norm[0], words_norm[1]
        if is_yandex_typical_error(w1, w2):
            return True, 'yandex_typical'
        if use_lemmatization and HAS_PYMORPHY and is_lemma_match(w1, w2):
            # v5.7.1: Не фильтровать если одно слово — это другое с приставкой "по-"
            # Пример: "больше"→"побольше" — реальная ошибка чтеца (глава 4 golden)
            is_po_prefix = False
            if w1.startswith('по') and len(w1) > 3 and w1[2:] == w2:
                is_po_prefix = True
            elif w2.startswith('по') and len(w2) > 3 and w2[2:] == w1:
                is_po_prefix = True
            if not is_po_prefix:
                return True, 'same_lemma'
        if is_yandex_name_error(w1, w2):
            return True, 'yandex_name_error'
        if len(w1) >= 5 and len(w2) >= 5 and levenshtein_distance(w1, w2) <= 1:
            is_meaningful_change = False
            if HAS_PYMORPHY:
                info1 = get_word_info(w1)
                info2 = get_word_info(w2)
                # info = (lemma, pos, number, gender, case)
                # Если одинаковая лемма — проверяем грамматику
                if info1[0] == info2[0]:
                    # Разное число — это реальная ошибка
                    if info1[2] and info2[2] and info1[2] != info2[2]:
                        is_meaningful_change = True
                    # v5.7.1: Разный падеж — это реальная ошибка (глава 4 golden)
                    # Пример: "преграды"→"преград", "награда"→"награды"
                    elif info1[4] and info2[4] and info1[4] != info2[4]:
                        is_meaningful_change = True
            if not is_meaningful_change:
                return True, 'levenshtein_protected'

    if has_protected:
        return False, 'protected_word'

    # ==== УРОВЕНЬ 2: Слабые слова / артефакты ====
    if error_type == 'deletion':
        word = words_norm[0]

        if word in ALIGNMENT_ARTIFACTS_DEL:
            return True, 'alignment_artifact'

        if word in SHORT_WEAK_WORDS:
            context = error.get('context', '')
            marker_pos = error.get('marker_pos', -1)
            if marker_pos > 0:
                before_context = context[:marker_pos].rstrip()
                if before_context and before_context[-1] not in '.!?':
                    return True, 'alignment_artifact'

        if word in WEAK_CONJUNCTIONS:
            context = error.get('context', '')
            marker_pos = error.get('marker_pos', -1)
            if marker_pos > 0:
                before_context = context[:marker_pos].rstrip()
                if before_context and before_context[-1] not in '.!?':
                    return True, 'alignment_artifact'
            elif marker_pos == 0:
                pass
            else:
                return True, 'alignment_artifact'

    if error_type == 'insertion':
        word = words_norm[0]
        if word in ALIGNMENT_ARTIFACTS_INS:
            return True, 'alignment_artifact'

        if word in WEAK_INSERTIONS:
            return True, 'alignment_artifact'

        # Паттерн: insertion и/а после конца предложения — это начало нового предложения
        # "довольно. И так", "Да. Они" — союз в начале предложения норма
        if word in {'и', 'а'}:
            context = error.get('context', '')
            marker_pos = error.get('marker_pos', -1)
            if marker_pos > 0 and context:
                before = context[:marker_pos].rstrip()
                if before and before[-1] in '.!?':
                    return True, 'sentence_start_conjunction'

        # ОТКЛЮЧЕНО v8.5.1: yandex_conjunction_before_gerund — см. deprecated_filters.py

        context = error.get('context', '').lower()
        if len(word) >= 3:
            context_words = context.split()
            for ctx_word in context_words:
                ctx_clean = normalize_word(ctx_word)
                if len(ctx_clean) > len(word) + 2 and word in ctx_clean:
                    return True, 'split_word_insertion'

    if error_type == 'substitution':
        w1, w2 = words_norm[0], words_norm[1]
        if all(w in weak_words for w in words_norm):
            if w1 == w2:
                return True, 'weak_words_identical'
            if HAS_PYMORPHY and is_lemma_match(w1, w2):
                return True, 'weak_words_same_lemma'

        # w1 = transcript (что Яндекс услышал)
        # w2 = original (что должно быть)

        # Паттерн 1: яХХ←я (Яндекс слил "я" со следующим словом)
        # оригинал "Я же" → транскрипт "яша"
        # w1="яша", w2="я"
        if len(w1) > 1 and w2 == 'я' and w1.startswith('я'):
            return True, 'yandex_merge_artifact'

        # Паттерн 2: и←их (Яндекс усёк многобуквенное до однобуквенного)
        # оригинал "Их главу" → транскрипт "И главу"
        # w1="и", w2="их"
        if w1 in {'и', 'а', 'я', 'е'} and len(w2) > 1 and w2.startswith(w1):
            return True, 'yandex_truncate_artifact'

        # Паттерн 3: итак←и (Яндекс расширил однобуквенное)
        # оригинал "И так" → транскрипт "итак"
        # w1="итак", w2="и"
        # v8.6.1: ИСКЛЮЧЕНИЕ для 'или' — это реальная ошибка чтеца (Golden)
        if w2 in {'и', 'а'} and len(w1) > 2 and w1.startswith(w2) and w1 not in {'или'}:
            return True, 'yandex_expand_artifact'

        # v5.6: Расширенный паттерн и↔я (→ v9.1 migrated to rules/)
        # Яндекс ОЧЕНЬ часто путает "и" и "я" в определённых контекстах
        if HAS_RULES_MODULE and ((w1 == 'и' and w2 == 'я') or (w1 == 'я' and w2 == 'и')):
            context = error.get('context', '')
            marker_pos = error.get('marker_pos', -1)
            should_filter, reason = check_i_ya_confusion(w1, w2, context, marker_pos)
            if should_filter:
                return True, reason

            # 6. Fallback: если pymorphy доступен, проверяем общий контекст
            # Если окружение содержит много глаголов — скорее всего ошибка Яндекса
            # v5.6.1: Увеличен порог с 2 до 3, чтобы избежать ложных срабатываний
            if HAS_PYMORPHY:
                context_lower = context.lower()
                context_words = context_lower.split()
                verb_count = 0
                for cw in context_words[:10]:  # Смотрим первые 10 слов
                    cw_clean = normalize_word(cw.rstrip('.,!?'))
                    if cw_clean and len(cw_clean) >= 2:
                        parsed = morph.parse(cw_clean)
                        if parsed and parsed[0].tag.POS in {'VERB', 'INFN', 'GRND'}:
                            verb_count += 1
                # Если много глаголов в контексте — скорее всего это ложное срабатывание
                if verb_count >= 3:
                    return True, 'yandex_i_ya_verb_context'

    # ==== УРОВЕНЬ 3: Только substitution ====
    if error_type == 'substitution':
        w1, w2 = words_norm[0], words_norm[1]

        if w1 == w2:
            return True, 'identical_normalized'

        if use_homophones and is_homophone_match(w1, w2):
            return True, 'homophone'

        if is_compound_word_match(w1, w2):
            return True, 'compound_word'

        original_context = error.get('context', '')
        if is_merged_word_error(w1, original_context):
            return True, 'merged_word'

        # v6.1: grammar_ending заменено на morpho_rules.py

        if is_case_form_match(w1, w2):
            return True, 'case_form'

        if is_adverb_adjective_match(w1, w2):
            return True, 'adverb_adjective'

        if is_short_full_adjective_match(w1, w2):
            return True, 'short_full_adjective'

        if is_verb_gerund_safe_match(w1, w2):
            return True, 'verb_gerund_safe'

        # v6.1: same_lemma и levenshtein заменены на morpho_rules.py

        if is_yandex_typical_error(w1, w2):
            return True, 'yandex_typical'

        if is_yandex_name_error(w1, w2):
            return True, 'yandex_name_error'

        # v6.1: prefix_variant заменён на morpho_rules.py

    # ==== УРОВЕНЬ 10: ML-классификатор v1.1 ====
    # Обучен на 93 golden + 648 FP. CV accuracy: 90%
    # Только substitution, порог 90% — консервативно
    # Тестировано: 'или→и' = REAL (59%), 'и→я' = REAL (82%)
    # v9.2.2: Защита от ML для грамматических ошибок (same_lemma + diff_number)
    # v9.3.2: Защита merged_from_ins_del — разные леммы = реальная ошибка
    if HAS_ML_CLASSIFIER and error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]

        # v9.3.2: Защита merged — разные леммы = реальная ошибка чтеца, ML не применяем
        skip_ml_merged = False
        if is_merged and HAS_PYMORPHY:
            lemma1_m = get_lemma(w1)
            lemma2_m = get_lemma(w2)
            if lemma1_m and lemma2_m and lemma1_m != lemma2_m:
                skip_ml_merged = True  # Merged с разными леммами — реальная ошибка

        if not skip_ml_merged:
            # v9.2.2: Защита — если одинаковая лемма, но разное число — это реальная ошибка
            # Пример: "идущими" vs "идущим" — ML ошибочно считает FP
            # v9.2.3: Дополнительно: если одинаковая лемма и разные окончания — потенциальная
            #         грамматическая ошибка (pymorphy может не различить число из-за омонимии)
            #         Пример: "тюрьмы" vs "тюрьма" — sing,gent vs plur,nomn — омонимия
            if HAS_PYMORPHY:
                lemma1 = get_lemma(w1)
                lemma2 = get_lemma(w2)
                num1 = get_number(w1)
                num2 = get_number(w2)
                if lemma1 and lemma2 and lemma1 == lemma2:
                    # Одинаковая лемма — проверяем грамматические различия
                    skip_ml = False
                    if num1 and num2 and num1 != num2:
                        # Явно разное число — реальная ошибка
                        skip_ml = True
                    elif w1 != w2 and len(w1) > 2 and len(w2) > 2:
                        # v9.2.3: Слова отличаются, но одинаковая лемма
                        # Это грамматическая вариация (падеж, число, род и т.д.)
                        # Не применяем ML — пусть человек проверит
                        skip_ml = True

                    if not skip_ml:
                        # Применяем ML только если слова идентичны (чистый FP)
                        try:
                            is_fp, confidence = _ml_classifier.predict(w1, w2)
                            if is_fp and confidence >= ML_CONFIDENCE_THRESHOLD:
                                return True, f'ml_classifier({confidence:.2f})'
                        except Exception:
                            pass
                else:
                    # Разные леммы — применяем ML
                    try:
                        is_fp, confidence = _ml_classifier.predict(w1, w2)
                        if is_fp and confidence >= ML_CONFIDENCE_THRESHOLD:
                            return True, f'ml_classifier({confidence:.2f})'
                    except Exception:
                        pass
            else:
                # Без морфологии — применяем ML
                try:
                    is_fp, confidence = _ml_classifier.predict(w1, w2)
                    if is_fp and confidence >= ML_CONFIDENCE_THRESHOLD:
                        return True, f'ml_classifier({confidence:.2f})'
                except Exception:
                    pass

    # ==== УРОВЕНЬ 11: SmartFilter — ОТКЛЮЧЁН ====
    # ПРИЧИНА: Все оставшиеся FP имеют score > 70 (grammar_change)
    # SmartFilter консервативен по дизайну — это ЗАЩИТА, не фильтрация
    # Результат: 0 дополнительных фильтраций — работает правильно
    # if HAS_SMART_FILTER and error_type == 'substitution' and len(words_norm) >= 2:
    #     try:
    #         smart_result = evaluate_error_smart(error, threshold=30)
    #         if smart_result and not smart_result.should_show:
    #             return True, f'smart_filter(score={smart_result.score})'
    #     except Exception:
    #         pass

    # ==== УРОВЕНЬ 12: ContextVerifier v1.0 ====
    # Контекстная верификация: проверяет артефакты склеенных/разбитых слов
    # Работает только для insertion (пока)
    # Безопасно: протестировано на golden, 0 ложных фильтраций
    if HAS_CONTEXT_VERIFIER and error_type == 'insertion':
        try:
            is_fp, reason = should_filter_by_context(error)
            if is_fp:
                return True, reason
        except Exception:
            pass

    return False, 'real_error'


def calculate_smart_score(error: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    v8.9: Рассчитывает SmartScore для ошибки (для метрик, не для фильтрации).

    Возвращает словарь с метриками или None если расчёт невозможен.
    Эти метрики добавляются к ошибке для:
    - Сортировки по приоритету (высокий score = важнее)
    - Аналитики и отладки
    - Будущей калибровки весов
    """
    if not HAS_SMART_SCORER:
        return None

    error_type = error.get('type', '')

    # Получаем слова
    if error_type == 'substitution':
        word1 = error.get('wrong', '') or error.get('transcript', '')
        word2 = error.get('correct', '') or error.get('original', '')
    elif error_type == 'insertion':
        word1 = error.get('wrong', '') or error.get('transcript', '') or error.get('word', '')
        word2 = ''
    elif error_type == 'deletion':
        word1 = ''
        word2 = error.get('correct', '') or error.get('original', '') or error.get('word', '')
    else:
        return None

    # Создаём скорер и результат
    scorer = SmartScorer()
    result = scorer.create_result(error_type, word2, word1)  # original, transcript

    # Базовый скоринг по типу
    scorer.apply_base_score(result)

    # Морфологический скоринг
    if HAS_PYMORPHY and error_type == 'substitution' and word1 and word2:
        w1_norm = normalize_word(word1)
        w2_norm = normalize_word(word2)

        lemma1 = get_lemma(w1_norm)
        lemma2 = get_lemma(w2_norm)
        pos1 = get_pos(w1_norm)
        pos2 = get_pos(w2_norm)

        same_lemma = lemma1 and lemma2 and lemma1 == lemma2
        same_pos = pos1 and pos2 and pos1 == pos2

        # Грамматические различия
        has_grammar_diff = False
        if same_lemma:
            num1 = get_number(w1_norm)
            num2 = get_number(w2_norm)
            case1 = get_case(w1_norm)
            case2 = get_case(w2_norm)
            if (num1 and num2 and num1 != num2) or (case1 and case2 and case1 != case2):
                has_grammar_diff = True

        scorer.apply_morphology(result, same_lemma, same_pos, has_grammar_diff)

        result.original_lemma = lemma2
        result.transcript_lemma = lemma1
        result.original_pos = pos2
        result.transcript_pos = pos1

    # Семантический скоринг
    if HAS_SEMANTIC_MANAGER and error_type == 'substitution' and word1 and word2:
        semantic_sim = get_similarity(word1, word2)
        scorer.apply_semantics(result, semantic_sim)

    return {
        'smart_score': result.score,
        'smart_rules': result.applied_rules,
        'is_visible': result.is_visible(),
        'original_lemma': result.original_lemma,
        'transcript_lemma': result.transcript_lemma,
        'original_pos': result.original_pos,
        'transcript_pos': result.transcript_pos,
        'semantic_similarity': result.semantic_similarity,
    }


def filter_errors(
    errors: List[Dict[str, Any]],
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """Фильтрует список ошибок."""
    filtered: List[Dict] = []
    removed: List[Dict] = []
    stats: Dict[str, int] = defaultdict(int)

    chain_indices = detect_alignment_chains(errors)
    linked_prefix_indices = detect_linked_prefix_errors(errors)

    for idx, error in enumerate(errors):
        if idx in chain_indices:
            stats['alignment_chain'] += 1
            removed.append({**error, 'filter_reason': 'alignment_chain'})
            continue

        if idx in linked_prefix_indices:
            stats['linked_prefix_error'] += 1
            removed.append({**error, 'filter_reason': 'linked_prefix_error'})
            continue

        should_filter, reason = should_filter_error(error, config, errors)
        stats[reason] += 1

        # v8.9: Добавляем SmartScore метрики к ошибке
        smart_metrics = calculate_smart_score(error)

        if should_filter:
            error_with_reason = {**error, 'filter_reason': reason}
            if smart_metrics:
                error_with_reason['smart_metrics'] = smart_metrics
            removed.append(error_with_reason)
        else:
            error_with_metrics = error.copy()
            if smart_metrics:
                error_with_metrics['smart_metrics'] = smart_metrics
            filtered.append(error_with_metrics)

    return filtered, removed, dict(stats)


def _validate_report_version(report: Dict[str, Any], report_path: str) -> None:
    """
    Проверяет версию compared.json на совместимость.
    Выдаёт предупреждение если файл создан устаревшей версией.
    """
    metadata = report.get('metadata', {})
    sc_version = metadata.get('smart_compare_version', 'unknown')

    if sc_version == 'unknown':
        print(f"  ⚠ ВНИМАНИЕ: {report_path}")
        print(f"    Файл не содержит метаданных версий.")
        print(f"    Рекомендуется пересоздать через smart_compare.py --force")
        return

    # Простое сравнение версий (формат X.Y.Z)
    def parse_version(v: str) -> tuple:
        try:
            parts = v.split('.')
            return tuple(int(p) for p in parts[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)

    current = parse_version(sc_version)
    minimum = parse_version(MIN_SMART_COMPARE_VERSION)

    if current < minimum:
        print(f"  ⚠ ВНИМАНИЕ: {report_path}")
        print(f"    Создан smart_compare v{sc_version}, требуется v{MIN_SMART_COMPARE_VERSION}+")
        print(f"    Рекомендуется пересоздать через smart_compare.py --force")


def filter_report(
    report_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    force: bool = False,
    skip_version_check: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Фильтрует отчёт с ошибками."""
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # v8.5: Валидация версии входного файла
    if not skip_version_check:
        _validate_report_version(report, report_path)

    errors = report.get('errors', [])
    original_count = len(errors)

    config: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    config.update(kwargs)

    try:
        from config import GoldenFilterConfig, FileNaming, check_file_exists
        HAS_CONFIG = True
        default_threshold = GoldenFilterConfig.LEVENSHTEIN_THRESHOLD
        default_lemma = GoldenFilterConfig.USE_LEMMATIZATION
        default_homophones = GoldenFilterConfig.USE_HOMOPHONES
    except ImportError:
        HAS_CONFIG = False
        default_threshold = 2
        default_lemma = True
        default_homophones = True

    print(f"\n{'='*60}")
    print(f"  Фильтр отсева v8.0 (morpho rules)")
    print(f"  Ошибок на входе: {original_count}")
    print(f"{'='*60}")
    print(f"  Настройки:")
    print(f"    Левенштейн порог: {config.get('levenshtein_threshold', default_threshold)}")
    print(f"    Лемматизация: {'да' if config.get('use_lemmatization', default_lemma) and HAS_PYMORPHY else 'нет'}")
    print(f"    Омофоны: {'да' if config.get('use_homophones', default_homophones) else 'нет'}")
    print(f"{'='*60}\n")

    filtered, removed, stats = filter_errors(errors, config)

    report['errors'] = filtered
    report['total_errors'] = len(filtered)
    report['filtered_count'] = original_count - len(filtered)
    report['filter_stats'] = stats

    cache_info = parse_word_cached.cache_info()
    # Добавляем метаданные фильтрации
    input_metadata = report.get('metadata', {})
    report['filter_metadata'] = {
        'version': VERSION,
        'input_smart_compare_version': input_metadata.get('smart_compare_version', 'unknown'),
        'input_alignment_manager_version': input_metadata.get('alignment_manager_version', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'original_errors': original_count,
        'real_errors': len(filtered),
        'filtered_errors': len(removed),
        'filter_efficiency': f"{(len(removed) / original_count * 100):.1f}%" if original_count > 0 else "0%",
        'cache_stats': {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'efficiency': f"{(cache_info.hits / (cache_info.hits + cache_info.misses) * 100):.1f}%" if (cache_info.hits + cache_info.misses) > 0 else "0%",
        },
        'filter_breakdown': {
            reason: {
                'count': count,
                'percentage': f"{(count / original_count * 100):.1f}%" if original_count > 0 else "0%",
            }
            for reason, count in sorted(stats.items(), key=lambda x: -x[1])
            if reason != 'real_error'
        },
        'error_types': {
            'substitution': len([e for e in filtered if e.get('type') == 'substitution']),
            'insertion': len([e for e in filtered if e.get('type') == 'insertion']),
            'deletion': len([e for e in filtered if e.get('type') == 'deletion']),
        },
    }

    # Сохраняем ВСЕ отфильтрованные ошибки для аудита
    report['filtered_errors_detail'] = removed

    if output_path:
        out_file = Path(output_path)
    else:
        if HAS_CONFIG:
            chapter_id = FileNaming.get_chapter_id(Path(report_path))
            out_file = Path(report_path).parent / FileNaming.build_filename(chapter_id, 'filtered')
        else:
            out_file = Path(report_path).with_stem(Path(report_path).stem + '_filtered')

    if out_file.exists() and not force:
        if HAS_CONFIG:
            check_file_exists(out_file, action='ask')
        else:
            print(f"  ⚠ Файл уже существует: {out_file.name}")

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  Результат:")
    print(f"    Реальных ошибок: {len(filtered)}")
    print(f"    Отфильтровано: {len(removed)}")
    print(f"\n  Причины фильтрации:")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        if reason != 'real_error':
            print(f"    {reason}: {count}")

    cache_info = parse_word_cached.cache_info()
    print(f"\n  Статистика кэша pymorphy:")
    print(f"    Попаданий: {cache_info.hits}")
    print(f"    Промахов: {cache_info.misses}")
    total_cache = cache_info.hits + cache_info.misses
    if total_cache > 0:
        print(f"    Эффективность: {cache_info.hits / total_cache * 100:.1f}%")

    print(f"\n  Сохранено: {out_file}")
    print(f"{'='*60}\n")

    return report
