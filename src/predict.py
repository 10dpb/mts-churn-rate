import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path


VERSION = "v1"
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / f"{VERSION}"
MODEL = MODEL_PATH / "cbm.pkl"
THRESHOLD_PATH = MODEL_PATH / "threshold.json"
TEST_CASE = PROJECT_DIR / "src" / "samples" / "test_sample.json"


def load_model_and_threshold():
    """Загружает модель и оптимальный порог"""
    model = joblib.load(MODEL)
    with open(THRESHOLD_PATH, 'r') as f:
        threshold_data = json.load(f)
    threshold = threshold_data['threshold']
    return model, threshold


def predict_from_json(json_file: Path):
    """
    Принимает JSON-файл с данными клиента (в формате, аналогичном обучению).
    Возвращает предсказание (0/1) и вероятность оттока.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    model, threshold = load_model_and_threshold()
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)

    results = []
    for i, row in df.iterrows():
        results.append({
            'probability_churn': round(proba[i], 4),
            'prediction': int(pred[i])
        })
    return results




if __name__ == "__main__":
    predictions = predict_from_json(TEST_CASE)
    print(json.dumps(predictions, indent=2))
    if predictions[0]["probability_churn"] == 0.0048:
        print("Тестирование пройдено успешно.")
