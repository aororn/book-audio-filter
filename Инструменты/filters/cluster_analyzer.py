#!/usr/bin/env python3
"""
Cluster Analyzer v1.0 — Анализ и фильтрация кластеров ошибок

Кластер — группа ошибок в пределах 2 сек друг от друга.
Многие FP являются артефактами выравнивания, которые проявляются как кластеры:
- substitution + insertion → split артефакт (слово разбилось)
- substitution + deletion → merge артефакт (слова слились)

Этот модуль:
1. Находит кластеры ошибок по времени
2. Определяет паттерны merge/split
3. Помечает артефакты для фильтрации

v1.0 (2026-01-31): Начальная версия
"""

VERSION = '1.0.0'

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class ErrorCluster:
    """Кластер связанных ошибок"""
    errors: List[Dict]
    chapter: int
    time_start: float
    time_end: float
    cluster_type: str  # merge_split, functional, other
    pattern: Optional[str] = None
    artifact_indices: List[int] = None  # Индексы ошибок-артефактов

    def __post_init__(self):
        if self.artifact_indices is None:
            self.artifact_indices = []


# Служебные слова, которые часто являются артефактами
FUNCTIONAL_WORDS = {
    'и', 'а', 'но', 'что', 'то', 'это', 'вот', 'ну', 'на',
    'он', 'она', 'они', 'я', 'вы', 'не', 'ни', 'же', 'бы',
    'ли', 'да', 'ей', 'их', 'до', 'при', 'за', 'от'
}

# Максимальное расстояние между ошибками в кластере (секунды)
CLUSTER_TIME_WINDOW = 2.0


def find_clusters(
    errors: List[Dict],
    time_window: float = CLUSTER_TIME_WINDOW,
    min_size: int = 2
) -> List[ErrorCluster]:
    """
    Находит кластеры ошибок по времени.

    Args:
        errors: Список ошибок (должны быть отсортированы по chapter, time_seconds)
        time_window: Максимальное расстояние между ошибками в кластере
        min_size: Минимальный размер кластера

    Returns:
        Список кластеров
    """
    if not errors:
        return []

    # Сортируем по главе и времени
    sorted_errors = sorted(errors, key=lambda e: (e.get('chapter', 0), e.get('time_seconds', 0)))

    clusters = []
    current_cluster_errors = []

    for error in sorted_errors:
        if not current_cluster_errors:
            current_cluster_errors = [error]
            continue

        last_error = current_cluster_errors[-1]
        same_chapter = error.get('chapter') == last_error.get('chapter')
        time_diff = abs(error.get('time_seconds', 0) - last_error.get('time_seconds', 0))

        if same_chapter and time_diff <= time_window:
            current_cluster_errors.append(error)
        else:
            if len(current_cluster_errors) >= min_size:
                cluster = _create_cluster(current_cluster_errors)
                clusters.append(cluster)
            current_cluster_errors = [error]

    # Последний кластер
    if len(current_cluster_errors) >= min_size:
        cluster = _create_cluster(current_cluster_errors)
        clusters.append(cluster)

    return clusters


def _create_cluster(errors: List[Dict]) -> ErrorCluster:
    """Создаёт объект кластера из списка ошибок"""
    chapter = errors[0].get('chapter', 0)
    times = [e.get('time_seconds', 0) for e in errors]

    # Определяем тип кластера
    types = [e.get('error_type', e.get('type', '')) for e in errors]
    has_sub = 'substitution' in types
    has_ins = 'insertion' in types
    has_del = 'deletion' in types

    if has_sub and (has_ins or has_del):
        cluster_type = 'merge_split'
    elif any(e.get('wrong', '').lower() in FUNCTIONAL_WORDS for e in errors):
        cluster_type = 'functional'
    else:
        cluster_type = 'other'

    return ErrorCluster(
        errors=errors,
        chapter=chapter,
        time_start=min(times),
        time_end=max(times),
        cluster_type=cluster_type
    )


def analyze_merge_split_pattern(cluster: ErrorCluster) -> Tuple[bool, str, List[int]]:
    """
    Анализирует кластер на наличие merge/split паттерна.

    Паттерны:
    1. Split: substitution(A→B) + insertion(C) где C+A = B или A+C = B
       Пример: "завтра"→"назавтра" + insertion("на") → "на" артефакт

    2. Merge: substitution(A→B) + deletion(C) где A = B+C
       Пример: "туда"→"то" + deletion("да") → "да" артефакт

    3. Duplicate: insertion(X) + deletion(X) — одно и то же слово
       Пример: insertion("нашли") + deletion("нашли") → оба артефакты

    4. Linked: ошибка уже помечена как linked (через error_links)
       → фильтруем связанные insertion/deletion

    Returns:
        (is_artifact, pattern_description, artifact_indices)
    """
    errors = cluster.errors

    # Получаем ошибки по типам
    types = [e.get('error_type', e.get('type', '')) for e in errors]
    has_sub = 'substitution' in types
    has_ins = 'insertion' in types
    has_del = 'deletion' in types

    # Если нет комбинации типов — не кластер для фильтрации
    if not ((has_sub and (has_ins or has_del)) or (has_ins and has_del)):
        return False, '', []

    subs = [(i, e) for i, e in enumerate(errors) if e.get('error_type', e.get('type', '')) == 'substitution']
    ins = [(i, e) for i, e in enumerate(errors) if e.get('error_type', e.get('type', '')) == 'insertion']
    dels = [(i, e) for i, e in enumerate(errors) if e.get('error_type', e.get('type', '')) == 'deletion']

    artifact_indices = []
    patterns = []

    # Проверяем split паттерны (substitution + insertion)
    for sub_idx, sub in subs:
        sub_wrong = (sub.get('wrong') or sub.get('transcript') or '').lower()
        sub_correct = (sub.get('correct') or sub.get('original') or '').lower()

        for ins_idx, ins_e in ins:
            ins_word = (ins_e.get('wrong') or ins_e.get('transcript') or '').lower()

            if not ins_word or not sub_correct:
                continue

            # Паттерн 1: inserted + wrong = correct (split)
            # Пример: "на" + "завтра" = "назавтра"
            if ins_word + sub_wrong == sub_correct:
                artifact_indices.append(ins_idx)
                patterns.append(f'split:{ins_word}+{sub_wrong}={sub_correct}')
                continue

            # Паттерн 2: wrong + inserted = correct (split)
            # Пример: "том" + "при" != "притом", но "при" + "том" = "притом"
            if sub_wrong + ins_word == sub_correct:
                artifact_indices.append(ins_idx)
                patterns.append(f'split:{sub_wrong}+{ins_word}={sub_correct}')
                continue

            # Паттерн 3: correct начинается с inserted
            # Пример: correct="назавтра", inserted="на"
            if sub_correct.startswith(ins_word) and len(ins_word) <= 3:
                remaining = sub_correct[len(ins_word):]
                if remaining == sub_wrong or _phonetic_similar(remaining, sub_wrong):
                    artifact_indices.append(ins_idx)
                    patterns.append(f'split_prefix:{ins_word}+{sub_wrong}≈{sub_correct}')
                    continue

            # Паттерн 4: correct заканчивается на inserted
            if sub_correct.endswith(ins_word) and len(ins_word) <= 3:
                remaining = sub_correct[:-len(ins_word)]
                if remaining == sub_wrong or _phonetic_similar(remaining, sub_wrong):
                    artifact_indices.append(ins_idx)
                    patterns.append(f'split_suffix:{sub_wrong}+{ins_word}≈{sub_correct}')
                    continue

            # Паттерн 4.1: correct = wrong + inserted (точное совпадение)
            # Пример: "таланты" = "талант" + "ы" (но фактически "талант" + "и")
            # Более мягкое: correct заканчивается на что-то похожее на inserted
            if len(ins_word) <= 2:
                suffix = sub_correct[-len(ins_word):] if len(sub_correct) >= len(ins_word) else ''
                prefix = sub_correct[:-len(ins_word)] if len(sub_correct) > len(ins_word) else sub_correct
                # wrong ≈ prefix от correct и inserted похоже на суффикс correct
                if _phonetic_similar(prefix, sub_wrong) and _phonetic_similar(suffix, ins_word):
                    artifact_indices.append(ins_idx)
                    patterns.append(f'split_suffix2:{sub_wrong}+{ins_word}→{sub_correct}')
                    continue

    # Проверяем merge паттерны (substitution + deletion)
    for sub_idx, sub in subs:
        sub_wrong = (sub.get('wrong') or sub.get('transcript') or '').lower()
        sub_correct = (sub.get('correct') or sub.get('original') or '').lower()

        for del_idx, del_e in dels:
            del_word = (del_e.get('correct') or del_e.get('original') or '').lower()

            if not del_word or not sub_wrong:
                continue

            # Паттерн 1: wrong = correct + deleted (merge)
            # Пример: "туда" = "то" + "да"
            if sub_correct + del_word == sub_wrong:
                artifact_indices.append(del_idx)
                patterns.append(f'merge:{sub_correct}+{del_word}={sub_wrong}')
                continue

            # Паттерн 2: wrong = deleted + correct (merge)
            if del_word + sub_correct == sub_wrong:
                artifact_indices.append(del_idx)
                patterns.append(f'merge:{del_word}+{sub_correct}={sub_wrong}')
                continue

            # Паттерн 3: wrong начинается с correct, deleted — окончание
            if sub_wrong.startswith(sub_correct) and sub_wrong.endswith(del_word):
                artifact_indices.append(del_idx)
                patterns.append(f'merge_overlap:{sub_correct}...{del_word}={sub_wrong}')
                continue

    # Паттерн 5: Дубликаты — insertion(X) + deletion(X)
    # Пример: insertion("нашли") + deletion("нашли") → оба артефакты
    for ins_idx, ins_e in ins:
        ins_word = (ins_e.get('wrong') or ins_e.get('transcript') or '').lower()
        for del_idx, del_e in dels:
            del_word = (del_e.get('correct') or del_e.get('original') or '').lower()

            if ins_word and del_word and ins_word == del_word:
                artifact_indices.append(ins_idx)
                artifact_indices.append(del_idx)
                patterns.append(f'duplicate:{ins_word}')

    # Паттерн 6: Linked ошибки (уже помечены через error_links)
    for i, e in enumerate(errors):
        linked = e.get('linked_errors')
        if linked:
            error_type = e.get('error_type', e.get('type', ''))
            # Фильтруем только insertion/deletion (substitution может быть golden)
            if error_type in ('insertion', 'deletion'):
                artifact_indices.append(i)
                patterns.append(f'linked:{error_type}')

    # Паттерн 7: Insertion служебного слова похожего на слово в substitution
    # Пример: ins("ну") + sub("но"→"а") — "ну" похоже на "но"
    for ins_idx, ins_e in ins:
        ins_word = (ins_e.get('wrong') or ins_e.get('transcript') or '').lower()
        if ins_word and ins_word in FUNCTIONAL_WORDS and len(ins_word) <= 3:
            for sub_idx, sub in subs:
                sub_correct = (sub.get('correct') or sub.get('original') or '').lower()
                # Если inserted похоже на correct из substitution
                if _phonetic_similar(ins_word, sub_correct) and len(sub_correct) <= 4:
                    artifact_indices.append(ins_idx)
                    patterns.append(f'func_phonetic:{ins_word}≈{sub_correct}')
                    break  # Уже нашли причину для этого insertion

    if artifact_indices:
        cluster.pattern = '; '.join(patterns)
        cluster.artifact_indices = list(set(artifact_indices))
        return True, cluster.pattern, cluster.artifact_indices

    return False, '', []


def _phonetic_similar(word1: str, word2: str, threshold: float = 0.7) -> bool:
    """Проверяет фонетическое сходство двух слов"""
    if not word1 or not word2:
        return False

    # Простое сравнение на основе общих символов
    if word1 == word2:
        return True

    # Нормализация
    w1 = word1.lower().replace('ё', 'е')
    w2 = word2.lower().replace('ё', 'е')

    if w1 == w2:
        return True

    # Фонетические эквиваленты (буквы, которые Яндекс часто путает)
    PHONETIC_GROUPS = [
        {'и', 'ы', 'й'},          # редуцированные гласные
        {'а', 'о'},               # безударные гласные
        {'е', 'э', 'и'},          # мягкие гласные
        {'у', 'ю'},               # огублённые
        {'б', 'п'},               # глухие-звонкие
        {'в', 'ф'},
        {'г', 'к', 'х'},
        {'д', 'т'},
        {'ж', 'ш', 'щ'},
        {'з', 'с'},
    ]

    def phonetic_normalize(char: str) -> str:
        """Нормализует символ к базовому в фонетической группе"""
        for group in PHONETIC_GROUPS:
            if char in group:
                return min(group)  # возвращаем первый в алфавитном порядке
        return char

    # Нормализуем обе строки
    norm1 = ''.join(phonetic_normalize(c) for c in w1)
    norm2 = ''.join(phonetic_normalize(c) for c in w2)

    if norm1 == norm2:
        return True

    # Для коротких слов (1-2 символа) — строгое сравнение после нормализации
    if len(w1) <= 2 and len(w2) <= 2:
        return norm1 == norm2

    # Для более длинных — допускаем частичное совпадение
    max_len = max(len(norm1), len(norm2))
    if max_len == 0:
        return True

    # Подсчёт общих символов
    common = sum(1 for c in norm1 if c in norm2)
    similarity = common / max_len

    return similarity >= threshold


def should_filter_by_cluster(
    error: Dict,
    all_errors: List[Dict],
    golden_ids: Set[str] = None
) -> Tuple[bool, str]:
    """
    Проверяет, должна ли ошибка быть отфильтрована как часть кластера.

    Args:
        error: Проверяемая ошибка
        all_errors: Все ошибки главы (для поиска кластера)
        golden_ids: Множество error_id золотых ошибок (для защиты)

    Returns:
        (should_filter, reason)
    """
    if golden_ids is None:
        golden_ids = set()

    # Находим кластеры
    clusters = find_clusters(all_errors)

    # Ищем кластер, содержащий нашу ошибку
    error_id = error.get('error_id', '')
    error_time = error.get('time_seconds', 0)
    error_chapter = error.get('chapter', 0)

    for cluster in clusters:
        # Проверяем, содержит ли кластер нашу ошибку
        cluster_error_ids = {e.get('error_id', '') for e in cluster.errors}
        cluster_times = {e.get('time_seconds', 0) for e in cluster.errors}

        in_cluster = (error_id and error_id in cluster_error_ids) or \
                     (error_time in cluster_times and error_chapter == cluster.chapter)

        if not in_cluster:
            continue

        # Анализируем паттерн
        is_artifact, pattern, artifact_indices = analyze_merge_split_pattern(cluster)

        if not is_artifact:
            continue

        # Находим индекс нашей ошибки в кластере
        for i, e in enumerate(cluster.errors):
            e_id = e.get('error_id', '')
            e_time = e.get('time_seconds', 0)

            matches = (error_id and error_id == e_id) or \
                      (error_time == e_time and error_chapter == e.get('chapter', 0))

            if matches and i in artifact_indices:
                # Проверяем, что ни одна ошибка в кластере не golden
                cluster_has_golden = any(
                    e.get('error_id', '') in golden_ids or e.get('is_golden', False)
                    for e in cluster.errors
                )

                if cluster_has_golden:
                    return False, ''

                return True, f'cluster_artifact:{pattern}'

    return False, ''


def get_cluster_artifacts(
    errors: List[Dict],
    golden_ids: Set[str] = None
) -> List[Tuple[Dict, str]]:
    """
    Возвращает список ошибок-артефактов кластеров с причинами.

    Args:
        errors: Все ошибки
        golden_ids: Множество error_id золотых ошибок

    Returns:
        Список (error, reason) для фильтрации
    """
    if golden_ids is None:
        golden_ids = set()

    artifacts = []
    clusters = find_clusters(errors)

    for cluster in clusters:
        # Проверяем, есть ли golden в кластере
        cluster_has_golden = any(
            e.get('error_id', '') in golden_ids or e.get('is_golden', False)
            for e in cluster.errors
        )

        if cluster_has_golden:
            continue

        is_artifact, pattern, artifact_indices = analyze_merge_split_pattern(cluster)

        if is_artifact:
            for idx in artifact_indices:
                error = cluster.errors[idx]
                reason = f'cluster_artifact:{pattern}'
                artifacts.append((error, reason))

    return artifacts


# Для тестирования
if __name__ == '__main__':
    import sqlite3
    import sys
    from pathlib import Path

    # Путь к БД
    db_path = Path(__file__).parent.parent.parent / 'Словари' / 'false_positives.db'

    if not db_path.exists():
        print(f"БД не найдена: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Загружаем все FP (включая linked_errors)
    cur.execute('''
        SELECT error_id, wrong, correct, error_type, chapter, time_seconds, is_golden, linked_errors
        FROM errors
        WHERE is_filtered = 0 AND is_golden = 0
        ORDER BY chapter, time_seconds
    ''')
    fp_errors = [dict(row) for row in cur.fetchall()]

    # Загружаем golden IDs
    cur.execute('SELECT error_id FROM errors WHERE is_golden = 1')
    golden_ids = {row[0] for row in cur.fetchall()}

    print(f"Cluster Analyzer v{VERSION}")
    print("=" * 60)
    print(f"Всего FP: {len(fp_errors)}")
    print(f"Golden IDs: {len(golden_ids)}")
    print()

    # Находим артефакты
    artifacts = get_cluster_artifacts(fp_errors, golden_ids)

    print(f"Найдено артефактов для фильтрации: {len(artifacts)}")
    print()

    for error, reason in artifacts[:20]:
        etype = error.get('error_type', '?')
        wrong = error.get('wrong', '?')
        correct = error.get('correct', '?')
        ch = error.get('chapter', '?')
        t = error.get('time_seconds', 0)
        print(f"  [{etype}] {wrong}→{correct} @{ch}/{int(t)//60}:{int(t)%60:02d}: {reason}")

    conn.close()
