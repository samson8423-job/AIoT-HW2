"""多元線性迴歸模型模組：Pipeline 建立、擬合、OLS 推論與預測區間。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from . import config
from .feature_selection import METHOD_SFS, make_selector


def count_transformed_features(preprocessor, X_train: pd.DataFrame, y_train: pd.Series) -> int:
    """在訓練資料上擬合前處理器，回傳轉換後（含 One-Hot）的特徵數量。"""
    fitted = clone(preprocessor).fit(X_train, y_train)
    return int(fitted.transform(X_train).shape[1])


def build_model_pipeline(
    preprocessor,
    method: str,
    k: int,
    direction: str,
    cv: int,
    n_transformed: int,
):
    """建立 前處理 -> 特徵選擇 -> LinearRegression 的 Pipeline。

    Returns:
        (pipeline, 實際使用的 K, 說明訊息列表)
    """
    notes: list[str] = []
    if n_transformed < 1:
        raise ValueError("轉換後沒有可用特徵，無法建立模型。")

    k_eff = max(1, min(int(k), n_transformed))
    if k_eff < int(k):
        notes.append(f"設定的 K={k} 大於可用特徵數 {n_transformed}，已自動調整為 {k_eff}。")

    if method == METHOD_SFS and k_eff >= n_transformed:
        k_eff = max(1, n_transformed - 1)
        notes.append(f"Sequential Feature Selection 需要少於全部特徵數，K 已調整為 {k_eff}。")

    selector, selector_notes = make_selector(
        method, k_eff, direction=direction, cv=cv, n_features=n_transformed
    )
    notes.extend(selector_notes)

    pipeline = Pipeline(
        [
            ("preprocess", clone(preprocessor)),
            ("selector", selector),
            ("model", LinearRegression()),
        ]
    )
    return pipeline, k_eff, notes


def selected_design_matrix(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """取得經前處理與特徵選擇後的設計矩陣（不含截距項），欄位名稱與 Pipeline 一致。"""
    transformed = pipeline[:-1].transform(X)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    support = np.asarray(pipeline.named_steps["selector"].get_support(), dtype=bool)
    return pd.DataFrame(np.asarray(transformed, dtype=float), columns=names[support])


def fit_ols(design: pd.DataFrame, y: pd.Series):
    """以 statsmodels OLS（含截距）在訓練設計矩陣上擬合，用於假設檢驗與預測區間。"""
    exog = sm.add_constant(design, has_constant="add")
    return sm.OLS(np.asarray(y, dtype=float), exog).fit()


def ols_exog(design: pd.DataFrame) -> np.ndarray:
    """與 fit_ols 相同欄位順序的解釋變數矩陣（含常數項）。"""
    return sm.add_constant(design, has_constant="add").to_numpy(dtype=float)


def prediction_interval(ols_results, design: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """計算個別新觀測值的預測區間（statsmodels get_prediction 的 obs_ci）。

    注意：obs_ci 描述個別觀測值的不確定性，比平均反應的信賴區間更寬。
    """
    frame = ols_results.get_prediction(ols_exog(design)).summary_frame(alpha=alpha)
    return frame[["mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper"]].reset_index(drop=True)


def model_coefficients(ols_results) -> pd.DataFrame:
    """OLS 係數表（含標準誤、t 統計量與 p-value）。"""
    names = list(ols_results.model.exog_names)
    return pd.DataFrame(
        {
            "變數": names,
            "係數": np.asarray(ols_results.params),
            "標準誤": np.asarray(ols_results.bse),
            "t 統計量": np.asarray(ols_results.tvalues),
            "p-value": np.asarray(ols_results.pvalues),
        }
    )
