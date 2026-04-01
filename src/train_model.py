import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
import catboost as cb
import optuna
import joblib
import json
import logging
import sys
from datetime import datetime
optuna.logging.set_verbosity(optuna.logging.WARNING)


VERSION = "v1"

# Пути
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "processed_data.csv"
MODEL_DIR = PROJECT_DIR / "models" / f"{VERSION}"
MODEL_PATH = MODEL_DIR / "cbm.pkl"
PARAMS_PATH = MODEL_DIR / "best_params.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
TEST_SCORES_PATH = MODEL_DIR / "test_with_scores.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Настройка логирования
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загружает обработанные данные"""
    logger.info(f"Загрузка данных из {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, target_col: str = "churn",
               test_size: float = 0.2, random_state: int = 42):
    """Разделяет данные на train и test с сохранением пропорций целевой переменной"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Разделение: train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"отток train={y_train.mean():.2%}, test={y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train, X_test, y_test, cat_features):
    """Целевая функция для Optuna"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'class_weights': {0: 1, 1: 20},
        'verbose': False,
        'random_seed': 42,
        'cat_features': cat_features,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'CPU'
    }
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=trial.suggest_int('early_stopping_rounds', 10, 50),
        verbose=False
    )
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    logger.debug(f"Trial {trial.number}: AUC={auc:.4f}, params={params}")
    return auc


def train_best_model(X_train, y_train, X_test, y_test, best_params, cat_features):
    """Обучает финальную модель с лучшими параметрами на всех тренировочных данных."""
    final_params = {k: v for k, v in best_params.items() if k not in ['verbose', 'eval_metric']}
    final_params['cat_features'] = cat_features
    final_params['loss_function'] = 'Logloss'
    final_params['eval_metric'] = 'AUC'
    final_params['verbose'] = 100

    model = cb.CatBoostClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=best_params.get('early_stopping_rounds', 30),
        verbose=100
    )
    return model


def find_optimal_threshold(y_true, y_proba, metric='f1', min_precision=None):
    """
    Находит оптимальный порог по уникальным значениям вероятностей.
    Если задан min_precision, выбирает порог с максимальным recall при precision >= min_precision.
    Иначе максимизирует выбранную метрику (f1, recall, precision).
    """
    unique_thresholds = np.unique(y_proba)
    # Добавляем 0 и 1 для краёв
    thresholds = np.sort(np.concatenate([unique_thresholds, [0, 1]]))
    best_threshold = 0.5
    best_score = -np.inf

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if min_precision is not None:
            precision = precision_score(y_true, y_pred, zero_division=0)
            if precision >= min_precision:
                recall = recall_score(y_true, y_pred, zero_division=0)
                score = recall
            else:
                continue
        else:
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    logger.info(f"Оптимальный порог: {best_threshold:.4f}, значение метрики: {best_score:.4f}")
    return best_threshold


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Оценивает модель с заданным порогом"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("=== Оценка модели ===")
    logger.info(f"ROC-AUC: {auc:.4f}")
    logger.info(f"Порог: {threshold:.4f}")
    logger.info(f"F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    logger.info("\nClassification Report:\n" + report)
    logger.info("\nConfusion Matrix:\n" + str(cm))
    return y_proba, y_pred, auc, f1, precision, recall, cm, report


def save_metrics(auc, f1, precision, recall, threshold, cm, report, params, path=METRICS_PATH):
    """Сохраняет метрики в JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dict = {
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "best_params": {k: v for k, v in params.items() if k not in ['verbose', 'eval_metric', 'cat_features']},
        "training_date": datetime.now().isoformat()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"Метрики сохранены в {path}")


def save_test_scores(X_test, y_test, y_proba, y_pred, path: Path = TEST_SCORES_PATH):
    """Сохраняет тестовый датасет с предсказаниями и вероятностями"""
    df_test = X_test.copy()
    df_test['true_churn'] = y_test.values
    df_test['pred_churn'] = y_pred
    df_test['probability_churn'] = y_proba
    df_test.to_csv(path, index=False)
    logger.info(f"Тестовый датасет со скорами сохранён в {path}")


def save_model(model, path: Path = MODEL_PATH):
    """Сохраняет модель с помощью joblib"""
    joblib.dump(model, path)
    logger.info(f"Модель сохранена в {path}")


def save_params(params, path: Path = PARAMS_PATH):
    """Сохраняет лучшие гиперпараметры в JSON"""
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)
    logger.info(f"Параметры сохранены в {path}")


def save_threshold(threshold, path: Path = THRESHOLD_PATH):
    """Сохраняет оптимальный порог"""
    with open(path, 'w') as f:
        json.dump({'threshold': threshold}, f)
    logger.info(f"Порог сохранён в {path}")


def main():
    logger.info("=== Запуск обучения модели ===")
    # Загрузка данных
    df = load_processed_data()
    logger.info(f"Загружено {df.shape[0]} записей, {df.shape[1]} признаков")
    logger.info(f"Доля оттока: {df['churn'].mean():.2%}")

    # Разделение
    X_train, X_test, y_train, y_test = split_data(df)

    # Категориальные признаки
    cat_features = X_train.select_dtypes(include=['string']).columns.tolist()
    
    # Оптимизация гиперпараметров
    logger.info("=== Оптимизация гиперпараметров через Optuna ===")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test, cat_features),
        n_trials=50,
        timeout=3600
    )
    best_params = study.best_params
    logger.info(f"Лучшие параметры: {best_params}")
    logger.info(f"Лучший ROC-AUC на валидации: {study.best_value:.4f}")

    # Обучение финальной модели
    logger.info("=== Обучение финальной модели ===")
    best_model = train_best_model(X_train, y_train, X_test, y_test, best_params, cat_features)

    # Подбор оптимального порога
    y_proba_val = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, y_proba_val, metric='f1')

    # Оценка с оптимальным порогом и получение вероятностей и метрик
    y_proba, y_pred, auc, f1, precision, recall, cm, report = \
        evaluate_model(best_model, X_test, y_test, threshold=optimal_threshold) 

    # Сохранение
    save_metrics(auc, f1, precision, recall, optimal_threshold, cm, report, best_params)
    save_test_scores(X_test, y_test, y_proba, y_pred)
    save_model(best_model)
    save_params(best_params)
    save_threshold(optimal_threshold)
    logger.info("=== Обучение завершено ===")




if __name__ == "__main__":
    main()