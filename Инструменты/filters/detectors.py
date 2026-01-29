"""
Специализированные детекторы для фильтрации ошибок транскрипции.

Содержит:
- load_character_names_dictionary — загрузка словаря имён
- is_yandex_name_error — ошибка Яндекса в имени персонажа
- is_merged_word_error — склеенные слова
- is_compound_word_match — слитное/раздельное написание
- is_split_name_insertion — разбитое имя (INS)
- is_compound_prefix_insertion — составное слово с префиксом (INS)
- is_split_compound_insertion — разбитое дефисное слово (INS)
- is_context_artifact — контекстные артефакты
- detect_alignment_chains — детектор цепочек смещения
"""

from pathlib import Path
from typing import Set, Dict, Any, List, Tuple, Optional

from .constants import (
    YANDEX_NAME_ERRORS, YANDEX_GARBLED_NAMES, CHARACTER_NAMES,
    SPLIT_NAME_PATTERNS, COMPOUND_PREFIXES, YANDEX_MERGED_WORDS,
    SPLIT_COMPOUND_PATTERNS,
)
from .comparison import (
    normalize_word, levenshtein_distance, HAS_PYMORPHY, morph,
)

# =============================================================================
# РАЗРЕШЕНИЕ ПУТЕЙ К СЛОВАРЯМ
# =============================================================================

try:
    from config import NAMES_DICT as _NAMES_DICT_PATH
except ImportError:
    _NAMES_DICT_PATH = Path(__file__).parent.parent / "Словари" / "Словарь_имён_персонажей.txt"


def _resolve_names_dict_path() -> Path:
    """Возвращает путь к словарю имён персонажей."""
    return _NAMES_DICT_PATH


# =============================================================================
# ЗАГРУЗКА СЛОВАРЯ ИМЁН
# =============================================================================

def load_character_names_dictionary() -> Set[str]:
    """Загружает словарь имён персонажей и генерирует все падежные формы."""
    names_set: Set[str] = set()

    dict_path = _resolve_names_dict_path()

    if not dict_path.exists():
        return names_set

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith('#'):
                continue

            names_set.add(name.lower())
            parts = name.split()
            for part in parts:
                names_set.add(part.lower())
                if HAS_PYMORPHY and morph:
                    try:
                        parsed = morph.parse(part)
                        if parsed:
                            for p in parsed[:3]:
                                for form in p.lexeme:
                                    names_set.add(form.word.lower())
                    except Exception:
                        pass

    return names_set


def load_base_character_names() -> Set[str]:
    """Загружает только базовые формы имён."""
    base_names: Set[str] = set()

    dict_path = _resolve_names_dict_path()

    if not dict_path.exists():
        return base_names

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith('#'):
                continue
            for part in name.split():
                base_names.add(part.lower())

    return base_names


# Загружаем словари при импорте
FULL_CHARACTER_NAMES = load_character_names_dictionary()
CHARACTER_NAMES_BASE = load_base_character_names()


# =============================================================================
# ДЕТЕКТОРЫ ИМЁН
# =============================================================================

def is_yandex_name_error(word1: str, word2: str) -> bool:
    """Ошибка Яндекса в имени персонажа."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if (w1, w2) in YANDEX_NAME_ERRORS or (w2, w1) in YANDEX_NAME_ERRORS:
        return True

    if w1 in YANDEX_GARBLED_NAMES or w2 in YANDEX_GARBLED_NAMES:
        return True

    if w1 in CHARACTER_NAMES_BASE or w2 in CHARACTER_NAMES_BASE:
        return True

    if FULL_CHARACTER_NAMES:
        w1_is_name = w1 in FULL_CHARACTER_NAMES
        w2_is_name = w2 in FULL_CHARACTER_NAMES

        if w1_is_name or w2_is_name:
            if w1_is_name and w2_is_name:
                return True

            if len(w1) >= 3 and len(w2) >= 3:
                dist = levenshtein_distance(w1, w2)
                max_len = max(len(w1), len(w2))
                if dist <= max(3, int(max_len * 0.5)):
                    return True

    return False


# =============================================================================
# ДЕТЕКТОРЫ СОСТАВНЫХ СЛОВ
# =============================================================================

def is_merged_word_error(word: str, context: str) -> bool:
    """Проверяет, является ли слово результатом склейки Яндексом."""
    word_norm = normalize_word(word)
    context_lower = context.lower()

    if word_norm in YANDEX_MERGED_WORDS:
        part1, part2 = YANDEX_MERGED_WORDS[word_norm]
        pattern = f'{part1} {part2}'
        if pattern in context_lower:
            return True

    return False


def is_compound_word_match(word1: str, word2: str) -> bool:
    """Слитное/раздельное написание."""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)

    if not w1 or not w2:
        return False

    short, long = (w1, w2) if len(w1) < len(w2) else (w2, w1)

    compound_patterns = [
        (short, short + 'то'), (short, short + 'нибудь'), (short, short + 'либо'),
        ('из', 'изза'), ('по', 'похорошему'), ('по', 'понастоящему'),
        ('не', 'недешевый'), ('не', 'неопасная'),
    ]

    for s, l in compound_patterns:
        if (short == s and long == l) or (w1 == s and w2 == l) or (w1 == l and w2 == s):
            return True

    # Исключаем пары "бы/было" — это разные слова с разным смыслом
    # "хотели бы" vs "хотели было" — совершенно разные конструкции
    if (short == 'бы' and long == 'было') or (short == 'было' and long == 'бы'):
        return False

    # Общее правило для составных слов (более строгое)
    # Требуем минимум 3 символа в коротком слове, чтобы избежать ложных срабатываний
    if len(short) >= 3 and long.startswith(short) and len(long) >= len(short) * 1.5:
        return True

    return False


def is_split_name_insertion(word: str, context: str) -> bool:
    """Проверяет, является ли вставка частью разбитого имени."""
    word_norm = normalize_word(word)
    context_lower = context.lower()

    for prefix, suffix in SPLIT_NAME_PATTERNS:
        if word_norm == suffix and prefix in context_lower:
            pattern = f'{prefix} {suffix}'
            if pattern in context_lower:
                return True

    return False


def is_compound_prefix_insertion(word: str, context: str) -> bool:
    """Проверяет, является ли вставка частью составного слова с префиксом."""
    word_norm = normalize_word(word)
    context_lower = context.lower()

    for prefix in COMPOUND_PREFIXES:
        pattern = f'{prefix} {word_norm}'
        if pattern in context_lower:
            return True

    return False


def is_split_compound_insertion(word: str, transcript_context: str, original_context: str = '') -> bool:
    """Проверяет, является ли вставка частью разбитого дефисного слова."""
    word_norm = normalize_word(word)
    transcript_lower = transcript_context.lower()
    original_lower = original_context.lower() if original_context else ''

    particles = {'то', 'либо', 'нибудь'}

    for prefix, suffix in SPLIT_COMPOUND_PATTERNS:
        if word_norm == suffix:
            if suffix in particles:
                compound = prefix + suffix
                if compound in original_lower:
                    return True
            else:
                pattern = f'{prefix} {suffix}'
                if pattern in transcript_lower:
                    return True

        if word_norm == prefix and suffix not in particles:
            pattern = f'{prefix} {suffix}'
            if pattern in transcript_lower:
                compound = prefix + suffix
                if compound in original_lower:
                    return True

    return False


def is_context_artifact(error: Dict[str, Any], errors_list: Optional[List[Dict]] = None) -> bool:
    """Контекстный фильтр."""
    context = error.get('context', '')
    error_type = error.get('type', '')

    if error_type == 'deletion':
        word = error.get('correct', '')
        word_norm = normalize_word(word)

        if word_norm == 'и':
            context_words = context.lower().split()
            i_count = context_words.count('и')
            if i_count >= 3:
                return True

    return False


# =============================================================================
# ДЕТЕКТОР ЦЕПОЧЕК СМЕЩЕНИЯ
# =============================================================================

def detect_alignment_chains(errors: List[Dict[str, Any]], time_window: float = 3.0) -> Set[int]:
    """
    Детектирует цепочки смещения — каскад ложных ошибок от сдвига выравнивания.
    Возвращает множество индексов ошибок, являющихся частью цепочки.
    """
    if not errors:
        return set()

    indexed_errors = [(i, e) for i, e in enumerate(errors)]
    indexed_errors.sort(key=lambda x: x[1].get('time', 0))

    chain_indices: Set[int] = set()

    for pos in range(len(indexed_errors) - 1):
        idx1, err1 = indexed_errors[pos]
        idx2, err2 = indexed_errors[pos + 1]

        time1 = err1.get('time', 0)
        time2 = err2.get('time', 0)

        if abs(time2 - time1) > time_window:
            continue

        wrong1 = normalize_word(err1.get('wrong', '') or err1.get('transcript', ''))
        correct1 = normalize_word(err1.get('correct', '') or err1.get('original', ''))
        wrong2 = normalize_word(err2.get('wrong', '') or err2.get('transcript', ''))
        correct2 = normalize_word(err2.get('correct', '') or err2.get('original', ''))

        is_shift = False
        if wrong1 and correct2 and wrong1 == correct2:
            is_shift = True
        if correct1 and wrong2 and correct1 == wrong2:
            is_shift = True

        if is_shift:
            if wrong1 and correct1:
                # ВАЖНО: Для очень коротких слов (1-2 буквы) не фильтруем —
                # это скорее реальная перестановка, чем артефакт выравнивания
                # Пример: "я и" → "и я" — реальная ошибка чтеца!
                min_len = min(len(wrong1), len(correct1))
                if min_len <= 2:
                    continue  # Короткие слова — не фильтруем

                dist = levenshtein_distance(wrong1, correct1)
                max_len = max(len(wrong1), len(correct1))
                if dist > 3 or (max_len > 0 and dist / max_len > 0.5):
                    chain_indices.add(idx1)
                    chain_indices.add(idx2)

                    for next_pos in range(pos + 2, len(indexed_errors)):
                        idx_next, err_next = indexed_errors[next_pos]
                        time_next = err_next.get('time', 0)

                        if abs(time_next - time2) > time_window:
                            break

                        wrong_next = normalize_word(err_next.get('wrong', '') or err_next.get('transcript', ''))
                        correct_next = normalize_word(err_next.get('correct', '') or err_next.get('original', ''))

                        prev_err = indexed_errors[next_pos - 1][1]
                        prev_wrong = normalize_word(prev_err.get('wrong', '') or prev_err.get('transcript', ''))
                        prev_correct = normalize_word(prev_err.get('correct', '') or prev_err.get('original', ''))

                        if (prev_wrong and correct_next and prev_wrong == correct_next) or \
                           (prev_correct and wrong_next and prev_correct == wrong_next):
                            chain_indices.add(idx_next)
                            time2 = time_next
                        else:
                            break

    return chain_indices


def detect_linked_prefix_errors(errors: List[Dict[str, Any]], time_window: float = 1.5) -> Set[int]:
    """
    Детектирует связанные ошибки типа ПРЕФИКС — когда Яндекс разбивает слово на два.

    Примеры:
    - "не молот" вместо "немолод" → substitution(немолод→не) + deletion(-молот)
    - "на встречу" вместо "навстречу" → substitution(навстречу→на) + deletion(-встречу)
    - "бес полной" вместо "без полной" → substitution(без→бесполной) + deletion(-полной)
    - "само возвышение" → deletion(-возвышение) при наличии "самовозвышение" рядом

    Возвращает множество индексов ошибок, которые являются частью связанной пары.
    """
    if not errors:
        return set()

    # Сортируем по времени
    indexed_errors = [(i, e) for i, e in enumerate(errors)]
    indexed_errors.sort(key=lambda x: x[1].get('time', 0))

    linked_indices: Set[int] = set()

    # Ищем пары ошибок, близкие по времени
    for pos in range(len(indexed_errors) - 1):
        idx1, err1 = indexed_errors[pos]
        idx2, err2 = indexed_errors[pos + 1]

        time1 = err1.get('time', 0)
        time2 = err2.get('time', 0)

        # Связанные ошибки обычно в пределах 1.5 секунд друг от друга
        if abs(time2 - time1) > time_window:
            continue

        type1 = err1.get('type', '')
        type2 = err2.get('type', '')

        # Получаем слова
        wrong1 = normalize_word(err1.get('wrong', '') or err1.get('transcript', ''))
        correct1 = normalize_word(err1.get('correct', '') or err1.get('original', ''))
        wrong2 = normalize_word(err2.get('wrong', '') or err2.get('transcript', ''))
        correct2 = normalize_word(err2.get('correct', '') or err2.get('original', ''))

        # Паттерн 1: substitution + deletion
        # Например: (немолод→не) + (deletion: -молот)
        # Яндекс услышал "немолод", оригинал "не молот" → две ошибки
        if type1 == 'substitution' and type2 == 'deletion':
            deleted_word = correct2  # слово, которое "пропущено"
            # Проверяем: можно ли объединить wrong1 с deleted_word?
            # Например: wrong1="немолод", deleted_word="молот" → "немолод" содержит "молот"
            if deleted_word and len(deleted_word) >= 3:
                # Вариант A: wrong1 заканчивается на deleted_word
                if wrong1.endswith(deleted_word) or wrong1.endswith(deleted_word[:-1]):
                    linked_indices.add(idx1)
                    linked_indices.add(idx2)
                    continue
                # Вариант B: wrong1 + ... = correct1 + deleted_word
                combined_orig = correct1 + deleted_word
                if levenshtein_distance(wrong1, combined_orig) <= 2:
                    linked_indices.add(idx1)
                    linked_indices.add(idx2)
                    continue

        # Паттерн 2: deletion + substitution (обратный порядок)
        if type1 == 'deletion' and type2 == 'substitution':
            deleted_word = correct1
            if deleted_word and len(deleted_word) >= 3:
                if wrong2.startswith(deleted_word) or wrong2.startswith(deleted_word[:-1]):
                    linked_indices.add(idx1)
                    linked_indices.add(idx2)
                    continue

        # Паттерн 3: две substitution, где вторая - часть первой
        # Например: (оголим→о) + (их→големах) — Яндекс разбил "о големах" на "оголим их"
        # v5.4: ужесточаем условия — хотя бы одно слово должно быть очень коротким (≤2)
        # Это значит, что Яндекс разбил длинное слово на частицу + слово
        if type1 == 'substitution' and type2 == 'substitution':
            # Признак разбиения: хотя бы одно из слов — короткая частица (о, и, а, я)
            # Если все слова длиной >= 3, это скорее всего независимые ошибки чтеца
            min_len = min(len(correct1), len(correct2), len(wrong1), len(wrong2))
            if min_len > 2:
                continue
            # Проверяем: correct1 + correct2 ≈ wrong1 + wrong2?
            combined_correct = correct1 + correct2
            combined_wrong = wrong1 + wrong2
            if len(combined_correct) >= 4 and len(combined_wrong) >= 4:
                if levenshtein_distance(combined_correct, combined_wrong) <= 3:
                    linked_indices.add(idx1)
                    linked_indices.add(idx2)
                    continue

    return linked_indices
