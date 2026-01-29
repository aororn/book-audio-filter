# План улучшения отсева ложных ошибок v11.2

**Дата:** 2026-01-30
**Текущий статус:** Golden 93/93, FP 268
**Цель:** FP ≤ 200 (уровень v5.7 = 183)

---

## 1. Текущее состояние БД

| Категория | Всего | Golden | FP | Avg Score |
|-----------|-------|--------|-----|-----------|
| grammar_ending | 321 | 34 | 287 | 60.0 |
| short_word | 238 | 22 | 216 | 20.5 |
| unknown | 144 | 19 | 125 | 59.8 |
| phonetic | 117 | 9 | 108 | 61.0 |
| prefix_variant | 34 | 1 | 33 | 63.6 |
| **ИТОГО** | **854** | **85** | **769** | — |

**ПРОБЛЕМА:** smart_score в БД рассчитаны по старым весам v2.0!

---

## 2. План итерационных тестов

### Этап 1: Пересчёт smart_score в БД (v3.0)

**Задача:** Обновить все smart_score с новыми весами SmartFilter v3.0.

```bash
python3 Инструменты/rebuild_smart_data.py --all
```

**Ожидаемый результат:**
- Golden: score >= 60 для всех 85
- FP: часть снизит score (станут фильтроваться)

### Этап 2: Анализ распределения после пересчёта

**Метрики для анализа:**
1. Сколько Golden теперь >= 60? (должно быть 100%)
2. Сколько FP теперь < 60? (потенциал фильтрации)
3. Какие FP по-прежнему >= 60? (требуют новых правил)

### Этап 3: Анализ проблемных FP

**Типы проблемных FP (score >= 60):**

| Тип | Примеры | Решение |
|-----|---------|---------|
| Имена персонажей | контуров→комтуров, шаугат→шугат | CharacterGuard: добавить в словарь |
| Опечатки Яндекса | калек→коллег, формации→информации | Levenshtein ≤ 2 + same_lemma → FP |
| Грамматические формы | силы→силой, нравится→нравятся | same_lemma + grammar_diff → FP для частых слов |

### Этап 4: Использование частотности

**Гипотеза:** Частые слова чаще имеют ошибки Яндекса (FP), редкие — реальные ошибки чтеца.

**Тесты:**
1. Для FP: если freq1 > 100 ipm AND same_lemma → вероятно FP
2. Для Golden: если freq1 < 10 ipm → важная ошибка (редкое слово)

**Данные НКРЯ:**
- 103K слов с частотностью
- Порог редкости: < 10 ipm
- Порог частоты: > 100 ipm

### Этап 5: Использование семантики

**Гипотеза:** Семантически близкие слова — вероятные оговорки (реальные ошибки).

**Тесты:**
1. Загрузить word2vec/fastText для русского
2. Рассчитать similarity для всех пар
3. Если similarity > 0.7 → semantic_slip → +30 баллов

**Данные:**
- RuWordNet или ru_core_news_lg (spaCy)

### Этап 6: Sliding Window для артефактов

**Гипотеза:** Артефакты выравнивания — когда контекст вокруг идентичен.

**Тесты:**
1. Проверять ±3 слова вокруг ошибки
2. Если совпадение > 90% → артефакт → -100 баллов
3. Анализ "склеек" типа "средо+точие"

---

## 3. Тестовый протокол

### Для каждого изменения:

1. **Пересчитать** smart_score в БД
2. **Проверить Golden:** `python3 Тесты/run_full_test.py --skip-pipeline`
3. **Подсчитать FP:** сумма total_errors по главам
4. **Записать результат** в таблицу

### Таблица результатов:

| Версия | Изменение | Golden | FP | Δ FP |
|--------|-----------|--------|-----|------|
| v11.1 | Baseline | 93/93 | 268 | — |
| v11.2a | Пересчёт БД | ?/93 | ? | ? |
| v11.2b | +CharacterNames | ?/93 | ? | ? |
| v11.2c | +Frequency rules | ?/93 | ? | ? |
| v11.2d | +Semantic slip | ?/93 | ? | ? |

---

## 4. Использование БД для анализа

### SQL-запросы для анализа:

```sql
-- 1. FP с высоким score (проблемные)
SELECT wrong, correct, category, smart_score, same_lemma
FROM patterns WHERE is_golden = 0 AND smart_score >= 60
ORDER BY count DESC LIMIT 50;

-- 2. Golden с низким score (риск фильтрации)
SELECT wrong, correct, category, smart_score, same_lemma
FROM patterns WHERE is_golden = 1 AND smart_score < 60
ORDER BY smart_score ASC;

-- 3. Кандидаты для CharacterGuard
SELECT wrong, correct, count
FROM patterns WHERE is_golden = 0
AND (wrong LIKE '%шауг%' OR wrong LIKE '%комтур%' OR wrong LIKE '%дараг%')
ORDER BY count DESC;

-- 4. Частые FP (same_lemma = 1)
SELECT wrong, correct, count, category
FROM patterns WHERE is_golden = 0 AND same_lemma = 1
ORDER BY count DESC LIMIT 30;

-- 5. Распределение по Levenshtein
SELECT levenshtein, COUNT(*) as cnt,
       SUM(CASE WHEN is_golden = 1 THEN 1 ELSE 0 END) as golden,
       SUM(CASE WHEN is_golden = 0 THEN 1 ELSE 0 END) as fp
FROM patterns GROUP BY levenshtein ORDER BY levenshtein;
```

---

## 5. Файлы для модификации

| Файл | Изменение |
|------|-----------|
| `filters/smart_scorer.py` | Новые веса |
| `filters/smart_filter.py` | Логика грамматики |
| `filters/frequency_manager.py` | Частотные правила |
| `filters/sliding_window.py` | Контекстная проверка |
| `Инструменты/rebuild_smart_data.py` | Пересчёт БД |
| `Словари/Словарь_имён_персонажей.txt` | Добавить имена |

---

## 6. Приоритеты

1. **HIGH:** Пересчёт smart_score в БД → проверка Golden
2. **HIGH:** Добавить имена персонажей в CharacterGuard
3. **MEDIUM:** Частотные правила для FP
4. **MEDIUM:** Семантическое сходство
5. **LOW:** Sliding Window (уже частично работает)

---

*Создан: 2026-01-30*
