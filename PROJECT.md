# Проверка аудиокниг через Яндекс SpeechKit

**Версия:** 14.15.0 | **Дата:** 2026-02-01

---

## Быстрый старт

```bash
source venv/bin/activate

# Пайплайн
python Инструменты/pipeline.py audio.mp3 original.docx

# Тесты
python Тесты/run_full_test.py
```

---

## Метрики

| Метрика | Значение |
|---------|----------|
| Golden | **127/127** ✓ |
| Ошибок | 388 (5 глав) |
| В БД | 1409 |
| Отфильтровано | 1021 |

---

## Архитектура

```
Аудио → Яндекс SpeechKit → Транскрипция
                               ↓
Оригинал → smart_compare.py → Фильтрация → SafetyVeto → Ошибки
                                   ↓
                              engine.py (10+ уровней)
                                   ↓
                              БД (автозапись)
```

### Слои фильтрации

```
L-0.6: Междометия
L-0.3: merge_artifact
L0: MorphoRules
L0.4-0.5: PhoneticSemantic
L0.6: AlignmentArtifacts
L1-9: Inline фильтры
L10: ContextVerifier (5 уровней)
L11: ML-классификатор (31 признак)
L13: ClusterAnalyzer
ФИНАЛ: SafetyVeto
```

---

## Ключевые модули

| Модуль | Версия | Назначение |
|--------|--------|------------|
| engine.py | 9.20 | Оркестратор фильтрации |
| error_normalizer.py | 1.0 | Унификация полей |
| context_verifier.py | 5.0 | 5 уровней верификации |
| safety_veto.py | 2.1 | Финальная защита |
| db_writer.py | 2.1 | Автозапись в БД |
| ml_classifier.py | 2.0 | ML (31 признак) |
| smart_compare.py | 10.6 | Выравнивание |

---

## База данных

```bash
python Инструменты/populate_db.py --stats   # статистика
python Инструменты/filters/db_writer.py --info  # инфо
```

Единый путь: `Словари/false_positives.db`

**Таблицы:** errors, error_history, sync_runs, error_links

---

## error_normalizer

Унификация полей (original↔correct, transcript↔wrong):

```python
from error_normalizer import (
    get_original_word,    # original/correct/from_book
    get_transcript_word,  # transcript/wrong/word
    errors_match,         # сравнение ошибок
)
```

---

## Принципы

1. **Консервативность** — при сомнении НЕ фильтруем
2. **Golden = 100%** — реальные ошибки не фильтруются
3. **Грамматика = ошибка** — разные леммы/падежи = реальная ошибка
4. **Унификация** — error_normalizer для консистентности

---

## Структура

```
Яндекс Спич/
├── Инструменты/
│   ├── pipeline.py, smart_compare.py, version.py
│   ├── error_normalizer.py, populate_db.py
│   └── filters/ (engine.py, db_writer.py, context_verifier.py, ...)
├── Тесты/ (run_full_test.py, золотой_стандарт_*.json)
├── Словари/ (false_positives.db)
├── Результаты проверки/ (01-05/)
└── PROJECT.md, ROADMAP.md, CHANGELOG.md
```

---

*v14.14.0 (2026-02-01)*
