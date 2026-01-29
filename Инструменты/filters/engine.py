"""
Движок фильтрации ошибок транскрипции v8.1.

Содержит:
- should_filter_error — решение по одной ошибке (20+ уровней фильтрации)
- filter_errors — фильтрация списка ошибок
- filter_report — фильтрация JSON-отчёта

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
from typing import Dict, List, Tuple, Optional, Any, Set

from .constants import (
    PROTECTED_WORDS, WEAK_WORDS,
    ALIGNMENT_ARTIFACTS_DEL, SHORT_WEAK_WORDS, WEAK_CONJUNCTIONS,
    ALIGNMENT_ARTIFACTS_INS, WEAK_INSERTIONS, FUNCTION_WORDS,
    YANDEX_SPLIT_INSERTIONS, SENTENCE_START_WEAK_WORDS,
    YANDEX_SPLIT_PAIRS, INTERROGATIVE_PRONOUNS, RARE_ADVERBS,
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

# v8.1: Импорт ScoringEngine для защиты имён и hard negatives
try:
    from .scoring_engine import (
        should_filter_by_score, is_hard_negative, HARD_NEGATIVES
    )
    HAS_SCORING_ENGINE = True
except ImportError:
    HAS_SCORING_ENGINE = False


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

    # ==== УРОВЕНЬ 0: Morpho Rules (v8.0) — консервативная фильтрация ====
    # Фильтруем ТОЛЬКО если 100% уверены в ложной ошибке
    # При любом грамматическом различии — НЕ фильтруем
    if error_type == 'substitution' and len(words_norm) >= 2:
        w1, w2 = words_norm[0], words_norm[1]

        morpho_result = get_morpho_rules().check(w1, w2)
        if morpho_result and morpho_result.should_filter:
            return True, f'morpho_{morpho_result.rule_name}'

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
                    if prefix.startswith('как') and next_word == 'там':
                        return True, 'compound_particle_to'
                    direction_words = {'туда', 'сюда', 'тут', 'здесь', 'теперь', 'тогда'}
                    verb_endings = ('ся', 'ет', 'ит', 'ут', 'ат', 'ют', 'ёт')
                    if next_word in direction_words or next_word.endswith(verb_endings):
                        continue
                return True, 'compound_particle_to'

    if error_type == 'insertion':
        transcript_context = error.get('transcript_context', '')
        if is_split_name_insertion(words_norm[0], transcript_context):
            return True, 'split_name'

    if error_type == 'insertion':
        transcript_context = error.get('transcript_context', '')
        if is_compound_prefix_insertion(words_norm[0], transcript_context):
            return True, 'compound_prefix'

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

    # INS из разбитых пар слов (жетон → "вот он", выторговали → "это говори")
    if error_type == 'insertion' and words_norm[0]:
        transcript_ctx = error.get('transcript_context', '').lower()
        inserted = words_norm[0]
        for (prev_word, ins_word), original_word in YANDEX_SPLIT_PAIRS.items():
            if inserted == ins_word:
                pattern = f'{prev_word} {ins_word}'
                if pattern in transcript_ctx:
                    return True, 'split_pair_yandex'

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

    # v5.3: INS дублирующиеся слова (где где, там там)
    # v5.5: Исправление — проверяем по отдельным словам, а не подстрокой
    if error_type == 'insertion' and words_norm[0]:
        inserted = words_norm[0]
        transcript_ctx = error.get('transcript_context', '').lower()
        trans_words = transcript_ctx.split()
        # Ищем два подряд идущих одинаковых слова (именно слова, не подстроки)
        for i in range(len(trans_words) - 1):
            if trans_words[i] == inserted and trans_words[i + 1] == inserted:
                return True, 'duplicate_word_insertion'

    # v5.3: INS коротких слов от разбиения длинных (мы от "големы", ли от "или")
    # v5.4: НЕ применяем к однобуквенным союзам/частицам — это настоящие ошибки чтеца
    SKIP_SPLIT_FRAGMENT = {'и', 'а', 'я', 'о', 'у', 'в', 'с', 'к'}
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

    # v5.5: DEL частиц "же"/"ли" в середине предложения
    # Яндекс часто пропускает эти частицы: "Я же просунул" → "Я просунул"
    # Но только если НЕ в начале предложения (там это может быть реальная ошибка)
    if error_type == 'deletion' and words_norm[0] in {'же', 'ли', 'ль'}:
        context = error.get('context', '')
        marker_pos = error.get('marker_pos', -1)
        if marker_pos > 0:
            before = context[:marker_pos].rstrip()
            # Проверяем: не в начале предложения (нет точки/вопроса/восклицания перед)
            if before and before[-1] not in '.!?':
                # Дополнительно: "же" после местоимения/существительного — частая ошибка Яндекса
                before_words = before.split()
                if before_words:
                    last_word = before_words[-1].lower().rstrip('.,!?')
                    # После местоимений: я же, он же, она же, они же, мы же, вы же, кто же
                    # После указательных: так же, тут же, то же
                    pronouns_and_adverbs = {
                        'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
                        'кто', 'что', 'это', 'то', 'так', 'тут', 'там', 'ещё', 'еще'
                    }
                    if last_word in pronouns_and_adverbs:
                        return True, 'yandex_particle_deletion'

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

        # v5.5: INS "и"/"а" перед деепричастием — Яндекс вставляет союз
        # Пример: "быстро, короткими фразами ведя" → "короткими фразами и ведя"
        # Паттерн: "и" перед словом на -я/-в/-ши/-вши (деепричастие)
        if word in {'и', 'а'} and HAS_PYMORPHY:
            transcript_ctx = error.get('transcript_context', '').lower()
            if transcript_ctx:
                trans_words = transcript_ctx.split()
                for i, tw in enumerate(trans_words):
                    if tw == word and i + 1 < len(trans_words):
                        next_word = trans_words[i + 1]
                        # Проверяем: следующее слово — деепричастие
                        parsed = morph.parse(next_word)
                        if parsed and parsed[0].tag.POS == 'GRND':
                            return True, 'yandex_conjunction_before_gerund'
                        # Резервный паттерн: окончания деепричастий
                        if next_word.endswith(('ая', 'яя', 'ив', 'ав', 'ши', 'вши')):
                            return True, 'yandex_conjunction_before_gerund'
                        break

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
        if w2 in {'и', 'а'} and len(w1) > 2 and w1.startswith(w2):
            return True, 'yandex_expand_artifact'

        # v5.6: Расширенный паттерн и↔я
        # Яндекс ОЧЕНЬ часто путает "и" и "я" в следующих контекстах:
        # 1. Граница предложений: "сказал. Я" → "сказал и"
        # 2. После глагола прошедшего времени: "кричал я" → "кричал и"
        # 3. После возвратного глагола: "справлюсь я" → "справлюсь и"
        # 4. Перед глаголом 1 лица: "я надеюсь" → "и надеюсь"
        # 5. В безударной позиции между двумя словами
        if (w1 == 'и' and w2 == 'я') or (w1 == 'я' and w2 == 'и'):
            context = error.get('context', '').lower()
            marker_pos = error.get('marker_pos', -1)

            if marker_pos > 0:
                before = context[:marker_pos].rstrip()
                after = context[marker_pos:].lstrip() if marker_pos < len(context) else ''
                # Убираем само слово из after (оно может быть частью контекста)
                after_words = after.split()[1:] if after.split() else []

                # Проверяем: заканчивается на глагол + точка или просто глагол
                before_words = before.replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').split()
                if before_words:
                    last_word = before_words[-1]
                    # 1. Если перед и/я стоит точка — граница предложений
                    if last_word in {'.', '!', '?'}:
                        return True, 'yandex_i_ya_boundary'
                    # 2. Если последнее слово — глагол прошедшего времени (-л/-ла/-ло/-ли)
                    if last_word.endswith(('л', 'ла', 'ло', 'ли', 'лся', 'лась', 'лось', 'лись')):
                        return True, 'yandex_i_ya_after_verb'
                    # 3. Возвратный глагол на -сь/-ся
                    if last_word.endswith(('сь', 'ся', 'шься', 'шись', 'юсь', 'усь')):
                        return True, 'yandex_i_ya_after_verb'
                    # 4. Глагол 1 лица на -у/-ю (думаю, верю, иду)
                    if last_word.endswith(('ю', 'у')) and len(last_word) >= 3:
                        return True, 'yandex_i_ya_after_verb'

                # 5. Перед глаголом 1 лица в будущем/настоящем времени
                if after_words:
                    next_word = after_words[0].rstrip('.,!?')
                    if next_word.endswith(('ю', 'у', 'юсь', 'усь')) and len(next_word) >= 3:
                        return True, 'yandex_i_ya_before_verb'
                    # Местоимения и наречия после я → реальное слово "я"
                    # Но Яндекс часто слышит "и" вместо "я" перед существительными
                    if HAS_PYMORPHY:
                        parsed = morph.parse(next_word)
                        if parsed and parsed[0].tag.POS in {'VERB', 'INFN'}:
                            # Если следующее слово — глагол, то я→и частая ошибка
                            return True, 'yandex_i_ya_before_verb'

            # 6. Fallback: если pymorphy доступен, проверяем общий контекст
            # Если окружение содержит много глаголов — скорее всего ошибка Яндекса
            # v5.6.1: Увеличен порог с 2 до 3, чтобы избежать ложных срабатываний
            if HAS_PYMORPHY:
                context_words = context.split()
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

        # v6.1: ОТКЛЮЧЕНО — заменено на smart_grammar
        # if is_grammar_ending_match(w1, w2):
        #     if HAS_PYMORPHY:
        #         lemma1 = get_lemma(w1)
        #         lemma2 = get_lemma(w2)
        #         if lemma1 == lemma2:
        #             pos1 = get_pos(w1)
        #             pos2 = get_pos(w2)
        #             if (pos1 == 'VERB' and pos2 == 'GRND') or (pos1 == 'GRND' and pos2 == 'VERB'):
        #                 pass
        #             else:
        #                 num1 = get_number(w1)
        #                 num2 = get_number(w2)
        #                 if num1 and num2 and num1 != num2:
        #                     pass
        #                 else:
        #                     return True, 'grammar_ending'

        if is_case_form_match(w1, w2):
            return True, 'case_form'

        if is_adverb_adjective_match(w1, w2):
            return True, 'adverb_adjective'

        if is_short_full_adjective_match(w1, w2):
            return True, 'short_full_adjective'

        if is_verb_gerund_safe_match(w1, w2):
            return True, 'verb_gerund_safe'

        # v6.1: ОТКЛЮЧЕНО — заменено на smart_lemma
        # if use_lemmatization and HAS_PYMORPHY and is_lemma_match(w1, w2):
        #     is_po_prefix = False
        #     if w1.startswith('по') and len(w1) > 3 and w1[2:] == w2:
        #         is_po_prefix = True
        #     elif w2.startswith('по') and len(w2) > 3 and w2[2:] == w1:
        #         is_po_prefix = True
        #     if not is_po_prefix:
        #         return True, 'same_lemma'

        # v6.1: ОТКЛЮЧЕНО — заменено на smart_levenshtein
        # if is_similar_by_levenshtein(w1, w2, levenshtein_threshold):
        #     ... (весь блок levenshtein_same_lemma и levenshtein_similar_lemma)

        if is_yandex_typical_error(w1, w2):
            return True, 'yandex_typical'

        if is_yandex_name_error(w1, w2):
            return True, 'yandex_name_error'

        # v6.1: ОТКЛЮЧЕНО — заменено на smart_prefix
        # if is_prefix_variant(w1, w2):
        #     return True, 'prefix_variant'

    return False, 'real_error'


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

        if should_filter:
            removed.append({**error, 'filter_reason': reason})
        else:
            filtered.append(error)

    return filtered, removed, dict(stats)


def filter_report(
    report_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    force: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Фильтрует отчёт с ошибками."""
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

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
    report['filter_metadata'] = {
        'version': '8.0.0',
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
