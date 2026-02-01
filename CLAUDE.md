# Lead-Dev-Partner v11.2

**Яндекс Спич v14.17.0** — Проверка аудиокниг через Yandex SpeechKit

---

## Роль

Ведущий разработчик: планирование → код → тесты → документация.

**Принципы:**
- Оценивай идеи: сложность, влияние на FP/golden, ресурсы
- Все архитектурные решения — только с подтверждением пользователя
- Golden 127/127 = 100% — ни одна реальная ошибка не фильтруется

---

## Метрики v14.17.0

| Метрика | Значение |
|---------|----------|
| Golden | **127/127** ✓ |
| Ошибок (5 глав) | 367 |
| В БД | 1409 |
| Отфильтровано | 1041 |

---

## Архитектура фильтрации

```
Вход → engine.py v9.20.1
│
├─ L-0.6: Междометия
├─ L-0.4: Whitelist FP-пар (v9.20.1)
├─ L-0.3: merge_artifact (ClusterAnalyzer)
├─ L0: MorphoRules
├─ L0.4-0.5: PhoneticSemantic v1.1
├─ L0.6: AlignmentArtifacts
├─ L1-9: Inline фильтры
├─ L10: ContextVerifier v6.0 (6 уровней)
├─ L11: ML-классификатор v2.0 (31 признак)
├─ L13: ClusterAnalyzer v1.1
├─ ФИНАЛ: SafetyVeto v2.2
│
└─ БД: populate_db v2.2 + error_history
```

---

## Ключевые модули

| Модуль | Версия | Назначение |
|--------|--------|------------|
| engine.py | 9.20.1 | Оркестратор фильтрации + whitelist |
| error_normalizer.py | 1.0 | Унификация полей (original↔correct) |
| context_verifier.py | 6.0 | 6 уровней контекстной верификации |
| safety_veto.py | 2.3 | Финальная защита VETO + имена персонажей |
| populate_db.py | 2.2 | БД + история изменений (error_history) |
| ml_classifier.py | 2.0 | ML (31 признак, порог 90%) |
| smart_compare.py | 10.6 | Посегментное выравнивание |

---

## Быстрые команды

```bash
source venv/bin/activate

# Тесты
python Тесты/run_full_test.py           # полный
python Тесты/run_full_test.py --skip-pipeline  # только golden

# Утилиты
python Инструменты/version.py           # версии
python Инструменты/populate_db.py --stats    # БД статистика
python Инструменты/populate_db.py --history  # История изменений
```

---

## Workflow

1. **Анализ** — найти паттерн в БД
2. **Реализация** — добавить в context_verifier.py (не в engine.py!)
3. **Тест** — `run_full_test.py --skip-pipeline`
4. **Документация** — CHANGELOG.md
5. **Версия** — инкремент в изменённых файлах + version.py

---

## Принципы

- **Контекстные фильтры** → context_verifier.py (изоляция)
- **error_normalizer** → унификация полей (original/transcript ↔ correct/wrong)
- **Консервативность** — при сомнении НЕ фильтруем
- **Грамматика = ошибка** — разные леммы/падежи = реальная ошибка чтеца

---

*v11.2 (2026-02-01) | Проект v14.17.0*
