#!/usr/bin/env python3
"""
ML Classifier v1.0 — Machine Learning классификатор для определения ложных срабатываний.

Использует морфологические признаки и метрики схожести для классификации
пар слов как реальных ошибок или ложных срабатываний.

Поддерживаемые модели:
- RandomForest (по умолчанию)
- GradientBoosting
- LogisticRegression

Использование:
    from ml_classifier import FalsePositiveClassifier

    clf = FalsePositiveClassifier()
    clf.train()  # Обучение на данных из false_positives.db
    clf.save()   # Сохранение модели

    # Предсказание
    is_fp, confidence = clf.predict("живем", "живы")

CLI:
    python ml_classifier.py train           # Обучить модель
    python ml_classifier.py predict w1 w2   # Предсказать
    python ml_classifier.py evaluate        # Оценить качество
    python ml_classifier.py info            # Информация о модели

СТАТУС: Опциональный инструмент (не интегрирован в основной пайплайн).
        Требует numpy + scikit-learn. Не влияет на golden тесты.

Changelog:
    v1.1 (2026-01-30): Унификация и документация
        - Унификация levenshtein и phonetic_normalize из filters.comparison
        - Добавлена документация статуса
    v1.0 (2026-01-26): Начальная версия
        - RandomForest/GradientBoosting/LogisticRegression
        - Признаки: Левенштейн, морфология, фонетика
        - Интеграция с false_positives_db.py
"""

VERSION = '1.1.0'
VERSION_DATE = '2026-01-30'
# v1.1: Унификация levenshtein и phonetic_normalize из filters.comparison

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import warnings

# Подавляем предупреждения sklearn
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# ИМПОРТ ЗАВИСИМОСТЕЙ
# =============================================================================

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠ numpy не установлен. Установите: pip install numpy")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠ scikit-learn не установлен. Установите: pip install scikit-learn")

# Импорт конфигурации
try:
    from config import TEMP_DIR, ML_MODEL_DIR
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    TEMP_DIR = Path(__file__).parent.parent / 'Темп'
    ML_MODEL_DIR = TEMP_DIR / 'ml'

# Импорт морфологии
try:
    from morphology import (
        get_lemma, get_pos, get_aspect, get_number, get_gender, get_case,
        normalize_word, HAS_PYMORPHY
    )
    HAS_MORPHOLOGY = True
except ImportError:
    HAS_MORPHOLOGY = False
    HAS_PYMORPHY = False

# Импорт фонетики и levenshtein (v1.1: унификация из comparison.py)
try:
    from filters.comparison import phonetic_normalize, levenshtein_distance as _levenshtein
    HAS_PHONETIC = True
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_PHONETIC = False
    HAS_LEVENSHTEIN = False
    # Fallback phonetic_normalize — базовая нормализация
    def phonetic_normalize(w):
        """Fallback: базовая русская фонетика."""
        w = w.lower().replace('ё', 'е')
        # Простые фонетические замены
        replacements = [
            ('тся', 'ца'), ('ться', 'ца'),
            ('чт', 'шт'), ('что', 'што'),
            ('его', 'ево'), ('ого', 'ово'),
        ]
        for old, new in replacements:
            w = w.replace(old, new)
        return w

# Импорт БД
try:
    from false_positives_db import FalsePositivesDB
    HAS_DB = True
except ImportError:
    HAS_DB = False


# =============================================================================
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ
# =============================================================================

# v1.1: levenshtein_distance унифицирован — используем из filters.comparison
def levenshtein_distance(s1: str, s2: str) -> int:
    """Расстояние Левенштейна (делегирует в filters.comparison)."""
    if HAS_LEVENSHTEIN:
        return _levenshtein(s1, s2)
    # Fallback — ручная реализация
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def extract_features(word1: str, word2: str) -> Dict[str, float]:
    """
    Извлекает признаки для пары слов.

    Returns:
        Словарь признаков
    """
    w1 = normalize_word(word1) if HAS_MORPHOLOGY else word1.lower()
    w2 = normalize_word(word2) if HAS_MORPHOLOGY else word2.lower()

    features = {}

    # 1. Длины слов
    features['len_w1'] = len(w1)
    features['len_w2'] = len(w2)
    features['len_diff'] = abs(len(w1) - len(w2))
    features['len_ratio'] = min(len(w1), len(w2)) / max(len(w1), len(w2)) if max(len(w1), len(w2)) > 0 else 1.0

    # 2. Левенштейн
    lev_dist = levenshtein_distance(w1, w2)
    features['levenshtein'] = lev_dist
    features['levenshtein_norm'] = lev_dist / max(len(w1), len(w2)) if max(len(w1), len(w2)) > 0 else 0

    # 3. Фонетика
    if HAS_PHONETIC:
        p1 = phonetic_normalize(w1)
        p2 = phonetic_normalize(w2)
        phon_dist = levenshtein_distance(p1, p2)
        features['phonetic_dist'] = phon_dist
        features['phonetic_same'] = 1.0 if p1 == p2 else 0.0
    else:
        features['phonetic_dist'] = lev_dist
        features['phonetic_same'] = 0.0

    # 4. Морфология
    if HAS_MORPHOLOGY:
        lemma1 = get_lemma(w1)
        lemma2 = get_lemma(w2)
        features['same_lemma'] = 1.0 if lemma1 == lemma2 else 0.0
        features['lemma_dist'] = levenshtein_distance(lemma1, lemma2)

        pos1 = get_pos(w1)
        pos2 = get_pos(w2)
        features['same_pos'] = 1.0 if pos1 == pos2 else 0.0

        # POS encoding (simplified)
        pos_map = {'NOUN': 1, 'VERB': 2, 'ADJF': 3, 'ADJS': 3, 'ADVB': 4, 'INFN': 2, 'GRND': 5, 'PRTF': 6, 'PRTS': 6}
        features['pos1_code'] = pos_map.get(pos1, 0)
        features['pos2_code'] = pos_map.get(pos2, 0)

        aspect1 = get_aspect(w1)
        aspect2 = get_aspect(w2)
        features['same_aspect'] = 1.0 if aspect1 == aspect2 else 0.0
        features['is_aspect_pair'] = 1.0 if (aspect1 and aspect2 and aspect1 != aspect2) else 0.0

        num1 = get_number(w1)
        num2 = get_number(w2)
        features['same_number'] = 1.0 if num1 == num2 else 0.0

        case1 = get_case(w1)
        case2 = get_case(w2)
        features['same_case'] = 1.0 if case1 == case2 else 0.0
    else:
        features['same_lemma'] = 0.0
        features['lemma_dist'] = lev_dist
        features['same_pos'] = 0.0
        features['pos1_code'] = 0
        features['pos2_code'] = 0
        features['same_aspect'] = 0.0
        features['is_aspect_pair'] = 0.0
        features['same_number'] = 0.0
        features['same_case'] = 0.0

    # 5. Общие паттерны
    # Одинаковое начало
    common_prefix = 0
    for a, b in zip(w1, w2):
        if a == b:
            common_prefix += 1
        else:
            break
    features['common_prefix'] = common_prefix
    features['common_prefix_ratio'] = common_prefix / max(len(w1), len(w2)) if max(len(w1), len(w2)) > 0 else 0

    # Одинаковый конец
    common_suffix = 0
    for a, b in zip(reversed(w1), reversed(w2)):
        if a == b:
            common_suffix += 1
        else:
            break
    features['common_suffix'] = common_suffix
    features['common_suffix_ratio'] = common_suffix / max(len(w1), len(w2)) if max(len(w1), len(w2)) > 0 else 0

    # Короткие слова (часто проблемные)
    features['is_short'] = 1.0 if min(len(w1), len(w2)) <= 3 else 0.0

    return features


def features_to_vector(features: Dict[str, float], feature_names: List[str]) -> List[float]:
    """Преобразует словарь признаков в вектор."""
    return [features.get(name, 0.0) for name in feature_names]


# =============================================================================
# ОСНОВНОЙ КЛАСС КЛАССИФИКАТОРА
# =============================================================================

@dataclass
class ModelInfo:
    """Информация о модели."""
    version: str
    model_type: str
    feature_names: List[str]
    train_samples: int
    train_accuracy: float
    cv_accuracy: float
    trained_at: str


class FalsePositiveClassifier:
    """
    ML-классификатор для определения ложных срабатываний.

    Обучается на данных из false_positives.db.
    """

    FEATURE_NAMES = [
        'len_w1', 'len_w2', 'len_diff', 'len_ratio',
        'levenshtein', 'levenshtein_norm',
        'phonetic_dist', 'phonetic_same',
        'same_lemma', 'lemma_dist', 'same_pos', 'pos1_code', 'pos2_code',
        'same_aspect', 'is_aspect_pair', 'same_number', 'same_case',
        'common_prefix', 'common_prefix_ratio',
        'common_suffix', 'common_suffix_ratio',
        'is_short'
    ]

    def __init__(self, model_type: str = 'random_forest', model_dir: Optional[Path] = None):
        """
        Инициализация.

        Args:
            model_type: 'random_forest', 'gradient_boosting', 'logistic_regression'
            model_dir: Директория для сохранения модели
        """
        if not HAS_SKLEARN or not HAS_NUMPY:
            raise ImportError("Требуются numpy и scikit-learn")

        self.model_type = model_type
        self.model_dir = model_dir or ML_MODEL_DIR
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.info: Optional[ModelInfo] = None

        # Создаём модель
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'  # Для несбалансированных классов
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает обучающие данные из БД.

        Returns:
            (X, y) — признаки и метки
        """
        if not HAS_DB:
            raise ImportError("false_positives_db.py недоступен")

        db = FalsePositivesDB()

        # Получаем все паттерны
        cursor = db.conn.execute('''
            SELECT wrong, correct, is_golden
            FROM patterns
            WHERE wrong IS NOT NULL AND correct IS NOT NULL
            AND LENGTH(wrong) > 0 AND LENGTH(correct) > 0
        ''')

        X_list = []
        y_list = []

        for row in cursor.fetchall():
            wrong = row[0]
            correct = row[1]
            is_golden = row[2]  # 1 = реальная ошибка, 0 = ложное срабатывание

            # Извлекаем признаки
            features = extract_features(wrong, correct)
            X_list.append(features_to_vector(features, self.FEATURE_NAMES))

            # Метка: 1 = ложное срабатывание (NOT golden), 0 = реальная ошибка (golden)
            y_list.append(0 if is_golden else 1)

        db.close()

        return np.array(X_list), np.array(y_list)

    def train(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Обучает модель на данных из БД.

        Args:
            test_size: Доля тестовой выборки

        Returns:
            Статистика обучения
        """
        from datetime import datetime

        print("Загрузка данных...")
        X, y = self._load_training_data()
        print(f"  Всего примеров: {len(y)}")
        print(f"  Реальных ошибок: {sum(y == 0)}")
        print(f"  Ложных срабатываний: {sum(y == 1)}")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Нормализация
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение
        print(f"\nОбучение {self.model_type}...")
        self.model.fit(X_train_scaled, y_train)

        # Оценка
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        cv_acc = cv_scores.mean()

        # Предсказания на тесте
        y_pred = self.model.predict(X_test_scaled)

        print(f"\n=== Результаты ===")
        print(f"Train accuracy: {train_acc:.2%}")
        print(f"Test accuracy: {test_acc:.2%}")
        print(f"CV accuracy: {cv_acc:.2%} (±{cv_scores.std():.2%})")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Реальная ошибка', 'Ложное сраб.']))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                   Predicted")
        print(f"                   Real   FP")
        print(f"Actual Real       {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"Actual FP         {cm[1][0]:4d}  {cm[1][1]:4d}")

        # Важность признаков (для RF/GB)
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nВажность признаков:")
            importances = list(zip(self.FEATURE_NAMES, self.model.feature_importances_))
            importances.sort(key=lambda x: -x[1])
            for name, imp in importances[:10]:
                print(f"  {name}: {imp:.3f}")

        # Сохраняем информацию о модели
        self.info = ModelInfo(
            version=VERSION,
            model_type=self.model_type,
            feature_names=self.FEATURE_NAMES,
            train_samples=len(y_train),
            train_accuracy=train_acc,
            cv_accuracy=cv_acc,
            trained_at=datetime.now().isoformat()
        )

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_accuracy': cv_acc,
            'cv_std': cv_scores.std(),
            'samples': len(y),
            'confusion_matrix': cm.tolist()
        }

    def predict(self, word1: str, word2: str) -> Tuple[bool, float]:
        """
        Предсказывает, является ли пара ложным срабатыванием.

        Args:
            word1: Первое слово (transcript/wrong)
            word2: Второе слово (original/correct)

        Returns:
            (is_false_positive, confidence)
        """
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите train() или load().")

        # Извлекаем признаки
        features = extract_features(word1, word2)
        X = np.array([features_to_vector(features, self.FEATURE_NAMES)])
        X_scaled = self.scaler.transform(X)

        # Предсказание
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]

        is_fp = bool(pred == 1)
        confidence = proba[pred]

        return is_fp, confidence

    def save(self, filename: str = 'fp_classifier.pkl') -> Path:
        """
        Сохраняет модель в файл.

        Returns:
            Путь к файлу
        """
        filepath = self.model_dir / filename

        data = {
            'model': self.model,
            'scaler': self.scaler,
            'info': self.info,
            'feature_names': self.FEATURE_NAMES,
            'version': VERSION
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Модель сохранена: {filepath}")
        return filepath

    def load(self, filename: str = 'fp_classifier.pkl') -> bool:
        """
        Загружает модель из файла.

        Returns:
            True если успешно
        """
        filepath = self.model_dir / filename

        if not filepath.exists():
            print(f"✗ Файл не найден: {filepath}")
            return False

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.info = data.get('info')

        print(f"✓ Модель загружена: {filepath}")
        if self.info:
            print(f"  Тип: {self.info.model_type}")
            print(f"  Обучено: {self.info.trained_at}")
            print(f"  CV accuracy: {self.info.cv_accuracy:.2%}")

        return True

    def get_info(self) -> Optional[ModelInfo]:
        """Возвращает информацию о модели."""
        return self.info


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_classifier_instance: Optional[FalsePositiveClassifier] = None


def get_classifier() -> FalsePositiveClassifier:
    """
    Возвращает глобальный экземпляр классификатора.
    Автоматически загружает модель если она существует.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FalsePositiveClassifier()
        # Пытаемся загрузить сохранённую модель
        try:
            _classifier_instance.load()
        except Exception:
            pass  # Модель не загружена, нужно обучить
    return _classifier_instance


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='ML Classifier для ложных срабатываний',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python ml_classifier.py train                    # Обучить модель
  python ml_classifier.py train --model gradient_boosting
  python ml_classifier.py predict живем живы      # Предсказать
  python ml_classifier.py evaluate                # Оценить качество
  python ml_classifier.py info                    # Информация о модели
        """
    )

    parser.add_argument('--version', '-V', action='store_true', help='Версия')

    subparsers = parser.add_subparsers(dest='command', help='Команда')

    # train
    train_parser = subparsers.add_parser('train', help='Обучить модель')
    train_parser.add_argument('--model', '-m', default='random_forest',
                             choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                             help='Тип модели')
    train_parser.add_argument('--test-size', '-t', type=float, default=0.2,
                             help='Доля тестовой выборки')

    # predict
    pred_parser = subparsers.add_parser('predict', help='Предсказать')
    pred_parser.add_argument('word1', help='Первое слово')
    pred_parser.add_argument('word2', help='Второе слово')

    # evaluate
    subparsers.add_parser('evaluate', help='Оценить качество')

    # info
    subparsers.add_parser('info', help='Информация о модели')

    # features
    feat_parser = subparsers.add_parser('features', help='Показать признаки пары слов')
    feat_parser.add_argument('word1', help='Первое слово')
    feat_parser.add_argument('word2', help='Второе слово')

    args = parser.parse_args()

    if args.version:
        print(f"ML Classifier v{VERSION} ({VERSION_DATE})")
        print(f"  numpy: {'да' if HAS_NUMPY else 'нет'}")
        print(f"  sklearn: {'да' if HAS_SKLEARN else 'нет'}")
        print(f"  morphology: {'да' if HAS_MORPHOLOGY else 'нет'}")
        return

    if not HAS_SKLEARN or not HAS_NUMPY:
        print("✗ Требуются numpy и scikit-learn")
        return

    if args.command == 'train':
        clf = FalsePositiveClassifier(model_type=args.model)
        stats = clf.train(test_size=args.test_size)
        clf.save()

    elif args.command == 'predict':
        clf = get_classifier()
        if clf.model is None:
            print("✗ Модель не обучена. Запустите: python ml_classifier.py train")
            return
        is_fp, conf = clf.predict(args.word1, args.word2)
        status = "✓ Ложное срабатывание" if is_fp else "✗ Реальная ошибка"
        print(f"{args.word1} / {args.word2}:")
        print(f"  {status}")
        print(f"  Уверенность: {conf:.1%}")

    elif args.command == 'evaluate':
        clf = get_classifier()
        if clf.model is None:
            print("✗ Модель не обучена. Запустите: python ml_classifier.py train")
            return
        # Просто перезапускаем train для оценки
        clf.train()

    elif args.command == 'info':
        clf = get_classifier()
        info = clf.get_info()
        if info:
            print(f"\n=== Информация о модели ===")
            print(f"Версия: {info.version}")
            print(f"Тип: {info.model_type}")
            print(f"Обучено на: {info.train_samples} примерах")
            print(f"Train accuracy: {info.train_accuracy:.2%}")
            print(f"CV accuracy: {info.cv_accuracy:.2%}")
            print(f"Обучено: {info.trained_at}")
            print(f"\nПризнаки ({len(info.feature_names)}):")
            for name in info.feature_names:
                print(f"  - {name}")
        else:
            print("✗ Модель не обучена")

    elif args.command == 'features':
        features = extract_features(args.word1, args.word2)
        print(f"\n=== Признаки: {args.word1} / {args.word2} ===\n")
        for name, value in features.items():
            print(f"  {name}: {value}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
