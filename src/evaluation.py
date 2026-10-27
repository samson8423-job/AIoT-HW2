"""模型評估模組：MAE、MSE、RMSE、R²、Adjusted R² 與比較表。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ADJ_R2_FORMULA = (
    "Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)，"
    "n 為測試集樣本數，p 為最終模型實際使用的轉換後特徵數（不含截距）。"
)

COMPARISON_COLUMNS = ["特徵選擇方法", "特徵數量", "MAE", "MSE", "RMSE", "R²", "Adjusted R²"]


def adjusted_r2(r2: float, n: int, p: int) -> tuple[float, str]:
    """計算 Adjusted R²。當 n ≤ p + 1 時回傳 NaN 與說明，不產生誤導性數值。"""
    if n <= p + 1:
        return float("nan"), f"不適用：樣本數 n={n} 不大於 p+1={p + 1}，無法可靠計算。"
    value = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return float(value), ADJ_R2_FORMULA


def regression_metrics(y_true, y_pred, n_predictors: int) -> dict:
    """以測試集實際預測結果計算五項指標。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    adj, adj_note = adjusted_r2(r2, n, int(n_predictors))
    return {
        "n_test": n,
        "p_predictors": int(n_predictors),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "R²": r2,
        "Adjusted R²": adj,
        "Adjusted R² 說明": adj_note,
    }


def comparison_table(method_results: list[dict]) -> pd.DataFrame:
    """將各方法的指標整理為比較表，預設依測試集 RMSE 由低至高排序。"""
    rows = []
    for item in method_results:
        metrics = item.get("metrics")
        if not metrics:
            continue
        rows.append(
            {
                "特徵選擇方法": item["method"],
                "特徵數量": item["n_selected"],
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R²": metrics["R²"],
                "Adjusted R²": metrics["Adjusted R²"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table.sort_values("RMSE", kind="stable").reset_index(drop=True)
