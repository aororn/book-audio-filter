# План тестирования и доработки SmartFilter v11

## Философия тестирования

### Golden ошибки (93 шт) — ЭТАЛОН
- Это 100% подтверждённые ошибки чтеца
- Алгоритмы должны их **НАХОДИТЬ**, а не защищать через хардкоды
- Если SmartFilter фильтрует golden → алгоритм неправильный
- Веса должны давать `score >= 60` для всех golden

### Ложные ошибки (FP) — 90% вероятность
- ~769 паттернов в БД с `is_golden = 0`
- 90% — реальные FP (ошибки Яндекса, артефакты)
- 10% — потенциальные пропуски (будущий модуль "YandexMissDetector")
- Веса должны давать `score < 60` для FP

---

## Текущее состояние БД (v3.0)

```
Всего паттернов: 854
  Golden: 85
  FP: 769

С actual_filter: 557 (65%)
Без actual_filter: 297 (35%) — цель для улучшения

Категории FP:
  grammar_ending: 321
  short_word: 238
  unknown: 144
  phonetic: 117
  prefix_variant: 34
```

---

## ЭТАП 1: Расширение схемы БД (v4.0)

### Новые колонки для SmartFilter

```sql
-- SmartFilter скоринг
ALTER TABLE patterns ADD COLUMN smart_score INTEGER;        -- итоговый скор
ALTER TABLE patterns ADD COLUMN smart_rules TEXT;           -- JSON: применённые правила
ALTER TABLE patterns ADD COLUMN smart_threshold INTEGER DEFAULT 60;

-- Частотный словарь
ALTER TABLE patterns ADD COLUMN freq1 REAL;                 -- частота wrong (ipm)
ALTER TABLE patterns ADD COLUMN freq2 REAL;                 -- частота correct (ipm)
ALTER TABLE patterns ADD COLUMN freq_category TEXT;         -- rare/bookish/common/unknown

-- Семантика
ALTER TABLE patterns ADD COLUMN semantic_similarity REAL;   -- косинус 0.0-1.0
ALTER TABLE patterns ADD COLUMN is_semantic_slip INTEGER;   -- 1 если sim > 0.6

-- SlidingWindow
ALTER TABLE patterns ADD COLUMN sliding_match INTEGER;      -- 1 если артефакт
ALTER TABLE patterns ADD COLUMN sliding_similarity INTEGER; -- 0-100

-- Валидация
ALTER TABLE patterns ADD COLUMN score_correct INTEGER;      -- 1 если score соответствует is_golden
```

---

## ЭТАП 2: Скрипт пересборки данных

### `rebuild_smart_data.py`

```python
"""
Пересчитать SmartFilter данные для всех паттернов в БД.
"""

def rebuild():
    db = FalsePositivesDB()
    sf = SmartFilter(use_semantics=True)
    fm = get_frequency_manager()
    sm = get_semantic_manager()  # загрузит 231 МБ модель
    sw = get_sliding_window()

    for pattern in db.get_all_patterns():
        # 1. Частотность
        freq1 = fm.get_frequency(pattern['wrong'])
        freq2 = fm.get_frequency(pattern['correct'])

        # 2. Семантика (только substitution)
        sim = 0.0
        if pattern['error_type'] == 'substitution':
            sim = sm.similarity(pattern['wrong'], pattern['correct'])

        # 3. SmartFilter скор
        error = {
            'type': pattern['error_type'],
            'wrong': pattern['wrong'],
            'correct': pattern['correct'],
            'context': pattern.get('context', ''),
        }
        result = sf.evaluate_error(error)

        # 4. Валидация
        is_golden = pattern['is_golden']
        score_correct = 1 if (result.should_show == (is_golden == 1)) else 0

        # 5. Обновляем БД
        db.update_smart_data(pattern['id'], {
            'smart_score': result.score,
            'smart_rules': json.dumps(result.applied_rules),
            'freq1': freq1,
            'freq2': freq2,
            'freq_category': result.frequency_category,
            'semantic_similarity': sim,
            'is_semantic_slip': 1 if sim > 0.6 else 0,
            'score_correct': score_correct,
        })
```

---

## ЭТАП 3: Тестовый фреймворк

### `test_smart_filter_accuracy.py`

```python
def test_golden_detection():
    """
    Все golden ошибки должны иметь score >= 60.
    """
    db = FalsePositivesDB()
    golden = db.get_patterns(is_golden=1)

    failures = []
    for g in golden:
        if g['smart_score'] < 60:
            failures.append({
                'pattern': g['pattern_key'],
                'score': g['smart_score'],
                'rules': g['smart_rules'],
            })

    assert len(failures) == 0, f"Golden errors filtered: {failures}"


def test_fp_filtering():
    """
    FP должны иметь score < 60 (цель: 70%+ FP Recall).
    """
    db = FalsePositivesDB()
    fps = db.get_patterns(is_golden=0)

    filtered = sum(1 for fp in fps if fp['smart_score'] < 60)
    recall = filtered / len(fps)

    print(f"FP Recall: {recall:.1%} ({filtered}/{len(fps)})")
    assert recall >= 0.70, f"FP Recall too low: {recall:.1%}"


def test_score_distribution():
    """
    Анализ распределения скоров для калибровки threshold.
    """
    db = FalsePositivesDB()

    # Golden распределение
    golden = db.get_patterns(is_golden=1)
    golden_scores = [g['smart_score'] for g in golden]

    # FP распределение
    fps = db.get_patterns(is_golden=0)
    fp_scores = [f['smart_score'] for f in fps]

    print(f"Golden scores: min={min(golden_scores)}, max={max(golden_scores)}, avg={sum(golden_scores)/len(golden_scores):.1f}")
    print(f"FP scores: min={min(fp_scores)}, max={max(fp_scores)}, avg={sum(fp_scores)/len(fp_scores):.1f}")

    # Оптимальный порог — максимальное разделение
    # Golden должны быть >= threshold
    # FP должны быть < threshold
```

---

## ЭТАП 4: Калибровка весов

### Метод калибровки

1. **Собрать данные**: Пересчитать скоры для всех 854 паттернов
2. **Анализ golden**: Какие правила срабатывают? Какие скоры?
3. **Анализ FP**: Какие правила дают высокий скор? Почему?
4. **Подбор весов**: Итеративно, чтобы golden >= 60, FP < 60

### Текущие веса (стресс-тест глава 1)

```python
WEIGHTS = {
    'character_shield': 100,  # имена = критично
    'deletion': 80,           # пропуск слова
    'substitution': 70,       # замена слова
    'insertion': 60,          # лишнее слово
    'pos_mismatch': 30,       # разная часть речи
    'different_lemma': 20,    # разные леммы
    'rare_word': 40,          # редкое слово (freq < 10)
    'bookish_word': 20,       # книжное (freq < 50)
    'semantic_synonym': 30,   # оговорка (sim > 0.6)
    'sliding_window_match': -100,  # артефакт
}
```

### Гипотезы для проверки

1. **Редкие слова важнее**: `rare_word: 40` может быть мало
   - Проверить: какие golden содержат редкие слова?
   - Проверить: какие FP содержат редкие слова? (это проблема!)

2. **Семантика помогает**: `semantic_synonym: 30`
   - Проверить: какие golden имеют высокий sim?
   - Гипотеза: оговорки чтеца ("способа"→"выхода") имеют sim > 0.6

3. **Sliding Window работает**: `-100` обнуляет артефакты
   - Проверить: не фильтрует ли golden?
   - Проверить: сколько FP ловит?

---

## ЭТАП 5: Итеративное улучшение

### Цикл разработки

```
1. Пересобрать SmartFilter данные
   python rebuild_smart_data.py

2. Запустить тесты
   pytest test_smart_filter_accuracy.py -v

3. Анализ провалов
   python analyze_failures.py

4. Корректировка весов
   # Изменить WEIGHTS в smart_scorer.py

5. Повторить с п.1
```

### Критерии успеха

| Метрика | Текущее | Цель |
|---------|---------|------|
| Golden Recall | 100% | 100% (нельзя терять) |
| FP Recall | 72.3% | ≥ 80% |
| FP Precision | N/A | ≥ 95% |

---

## ЭТАП 6: Анализ провалов

### `analyze_failures.py`

```python
def analyze_golden_filtered():
    """
    Почему golden был отфильтрован?
    """
    db = FalsePositivesDB()
    golden = db.get_patterns(is_golden=1)

    for g in golden:
        if g['smart_score'] < 60:
            print(f"\n=== {g['pattern_key']} ===")
            print(f"Score: {g['smart_score']}")
            print(f"Rules: {g['smart_rules']}")
            print(f"Freq: {g['freq1']} → {g['freq2']}")
            print(f"Semantic: {g['semantic_similarity']}")
            print(f"Category: {g['category']}")


def analyze_fp_not_filtered():
    """
    Почему FP не был отфильтрован?
    """
    db = FalsePositivesDB()
    fps = db.get_patterns(is_golden=0)

    high_score_fps = [f for f in fps if f['smart_score'] >= 60]
    print(f"FP с высоким скором: {len(high_score_fps)}")

    # Группируем по причине
    by_reason = {}
    for fp in high_score_fps:
        rules = json.loads(fp['smart_rules'] or '[]')
        key = ', '.join(rules) if rules else 'no_rules'
        by_reason[key] = by_reason.get(key, 0) + 1

    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")
```

---

## Файлы для создания

| Файл | Назначение |
|------|------------|
| `false_positives_db.py` v4.0 | + колонки SmartFilter |
| `rebuild_smart_data.py` | Пересборка данных |
| `test_smart_filter_accuracy.py` | Тесты точности |
| `analyze_failures.py` | Анализ провалов |
| `calibrate_weights.py` | Подбор весов |

---

## Следующие шаги

1. [ ] Добавить колонки SmartFilter в БД (миграция v4)
2. [ ] Создать `rebuild_smart_data.py`
3. [ ] Пересобрать данные для 854 паттернов
4. [ ] Запустить `test_smart_filter_accuracy.py`
5. [ ] Анализ и калибровка весов
6. [ ] Достичь FP Recall ≥ 80%

---

*План создан: 2026-01-30*
*SmartFilter v1.0, БД v3.0*
