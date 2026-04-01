import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
LOAD_PATH = PROJECT_DIR / "data" / "raw" / "telco-customer-churn.csv"
SAVE_PATH = PROJECT_DIR / "data" / "processed" / "processed_data.csv"


def load_data(filepath: Path = LOAD_PATH) -> pd.DataFrame:
    """Загружает исходные данные"""
    return pd.read_csv(filepath)


def aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует помесячные данные до уровня клиента.
    Для статичных признаков берёт первое значение.
    Для динамических числовых признаков создаёт mean, std, min, max
    """
    
    # Статичные признаки (после удаления неинформативных)
    static_cols = [
        "age",
        "gender",
        "maritalstatus",
        "education",
        "occupation",
        "annualincome",
        "homeowner",
        "state",
        "monthlybilledamount",
        "totalminsusedinlastmonth",
        "unpaidbalance",
        "numberofmonthunpaid",
        "numberofcomplaints",
        "numdayscontractequipmentplanexpiring",
        "penaltytoswitch",
        "calldroprate",
        "callfailurerate",
        "percentagecalloutsidenetwork",
    ]

    # Динамические числовые признаки
    dynamic_num_cols = ["totalcallduration", "avgcallduration"]

    # Словарь агрегации
    agg_dict = {col: "first" for col in static_cols if col in df.columns}
    for col in dynamic_num_cols:
        if col in df.columns:
            agg_dict[col] = ["mean", "std", "min", "max"]
    agg_dict["churn"] = "max"

    df_agg = df.groupby("customerid").agg(agg_dict)
    # Разворачиваем мультииндекс в плоские имена
    df_agg.columns = [
        '_'.join(col).strip('_')
        if col[1] != "first" else col[0]
        for col in df_agg.columns.values
    ]
    df_agg.reset_index(inplace=True)
    df_agg.drop(columns=["customerid"], inplace=True)
    df_agg.rename(columns={"churn_max": "churn"}, inplace=True)
    print(f"Размер после агрегации: {df_agg.shape}")
    return df_agg


def save_data(df: pd.DataFrame, path: Path = SAVE_PATH):
    """Сохраняет обработанные данные"""
    df.to_csv(path, index=False)
    print(f"Обработанные данные сохранены в {str(path)}")


def prepare_dataset(path_from: Path = LOAD_PATH, path_to: Path = SAVE_PATH):
    """
    Основная функция подготовки данных.
    Загружает, очищает, агрегирует, создаёт признаки и сохраняет результат.
    """
    df = load_data(path_from)
    df_agg = aggregate_customer_data(df)
    save_data(df_agg, path_to)




if __name__ == "__main__":
    prepare_dataset()