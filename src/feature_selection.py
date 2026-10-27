"""五種特徵選擇方法。

所有選擇器都放在 Pipeline 中，且只在訓練資料上 fit，避免測試集資訊洩漏。
每個選擇器都提供 get_support()，讓後續可統一取得被選中的轉換後特徵。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from . import config

METHOD_PEARSON = "Pearson 相關係數"
METHOD_KBEST = "SelectKBest (f_regression)"
METHOD_RFE = "RFE (LinearRegression)"
METHOD_SFS = "Sequential Feature Selection"
METHOD_LASSO = "LassoCV (L1 正則化)"

METHODS = [METHOD_PEARSON, METHOD_KBEST, METHOD_RFE, METHOD_SFS, METHOD_LASSO]
METHOD_DESCRIPTIONS = {
    METHOD_PEARSON: "依訓練資料中每個轉換後特徵與目標變數的 Pearson 相關係數絕對值排序，取前 K 個。",
    METHOD_KBEST: "以 f_regression 計算每個特徵與目標的線性 F 分數，取分數最高的前 K 個。",
    METHOD_RFE: "以 LinearRegression 的係數大小反覆移除最不重要的特徵，直到剩下 K 個。",
    METHOD_SFS: "以 LinearRegression 搭配交叉驗證（RMSE），逐步加入（前向）或移除（後向）特徵，直到剩下 K 個。",
    METHOD_LASSO: "以 LassoCV（交叉驗證選 alpha）在標準化後的特徵上擬合，保留係數非零的特徵，特徵數由資料決定。",
}


METHOD_SHORT = {
    METHOD_PEARSON: "Pearson",
    METHOD_KBEST: "SelectKBest",
    METHOD_RFE: "RFE",
    METHOD_SFS: "SFS",
    METHOD_LASSO: "Lasso",
}


def _pearson_with_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐欄計算 Pearson 相關係數；標準差為零的欄位回傳 0。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt((x_centered**2).sum(axis=0)) * np.sqrt((y_centered**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, 0.0)


class PearsonSelector(BaseEstimator, TransformerMixin):
    """依 |Pearson r| 排序，保留前 k 個特徵（僅使用傳入的訓練資料）。"""

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("PearsonSelector 需要目標變數 y。")
        X = np.asarray(X, dtype=float)
        r = _pearson_with_target(X, y)
        order = np.argsort(-np.abs(r), kind="stable")
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[order[: min(self.k, X.shape[1])]] = True
        self.scores_ = r
        self.support_ = mask
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.where(self.support_)[0] if indices else self.support_


class LassoSelector(BaseEstimator, TransformerMixin):
    """以 StandardScaler + LassoCV 選擇非零係數的特徵。

    若所有係數皆為零，則保留與目標相關係數絕對值最大的一個特徵，並將 used_fallback_ 設為 True，
    以便介面明確提示此情況。
    """

    def __init__(self, cv: int = 5):
        self.cv = cv

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LassoSelector 需要目標變數 y。")
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.lasso_ = LassoCV(cv=self.cv, max_iter=20000)
        self.lasso_.fit(X_scaled, y)
        self.coef_ = self.lasso_.coef_
        mask = np.abs(self.coef_) > 1e-10
        self.used_fallback_ = False
        if not mask.any():
            r = _pearson_with_target(X, y)
            mask[int(np.argmax(np.abs(r)))] = True
            self.used_fallback_ = True
        self.support_ = mask
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return np.where(self.support_)[0] if indices else self.support_


def make_selector(
    method: str,
    k: int,
    direction: str = "forward",
    cv: int = config.DEFAULT_CV_FOLDS,
    n_features: int | None = None,
):
    """建立指定方法的特徵選擇器，回傳 (選擇器, 說明訊息列表)。"""
    notes: list[str] = []
    if method == METHOD_PEARSON:
        return PearsonSelector(k=k), notes
    if method == METHOD_KBEST:
        return SelectKBest(score_func=f_regression, k=k), notes
    if method == METHOD_RFE:
        return RFE(LinearRegression(), n_features_to_select=k, step=1), notes
    if method == METHOD_SFS:
        chosen_direction = direction
        if (
            direction == "backward"
            and n_features is not None
            and n_features > config.BACKWARD_SFS_MAX_FEATURES
        ):
            chosen_direction = "forward"
            notes.append(
                f"轉換後特徵數為 {n_features}，超過後向選擇上限 {config.BACKWARD_SFS_MAX_FEATURES}，"
                "為避免運算時間過長，已自動改用前向選擇。"
            )
        selector = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction=chosen_direction,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        return selector, notes
    if method == METHOD_LASSO:
        return LassoSelector(cv=cv), notes
    raise ValueError(f"未知的特徵選擇方法：{method}")


def selection_detail(pipeline, method: str) -> tuple[pd.DataFrame, str]:
    """整理被擬合後的選擇器結果。

    Returns:
        (明細表, 分數欄位的說明)。明細表欄位：轉換後特徵、分數、是否選中。
    """
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    selector = pipeline.named_steps["selector"]
    support = np.asarray(selector.get_support(), dtype=bool)

    if method == METHOD_PEARSON:
        score = selector.scores_
        label = "Pearson r"
    elif method == METHOD_KBEST:
        score = selector.scores_
        label = "F 分數"
    elif method == METHOD_RFE:
        score = selector.ranking_.astype(float)
        label = "RFE 排名（1 為最重要）"
    elif method == METHOD_LASSO:
        score = selector.coef_
        label = "標準化後的 Lasso 係數"
    else:
        score = np.full(len(feature_names), np.nan)
        label = "SFS 未提供單一分數（僅顯示是否選中）"

    detail = pd.DataFrame(
        {
            "轉換後特徵": feature_names,
            "分數": np.round(np.asarray(score, dtype=float), 6),
            "是否選中": np.where(support, "是", "否"),
        }
    )
    if method == METHOD_KBEST:
        detail["p-value"] = np.round(selector.pvalues_, 6)
    if method == METHOD_RFE:
        detail = detail.sort_values("分數", kind="stable").reset_index(drop=True)
    elif method == METHOD_LASSO:
        detail["絕對係數"] = np.round(np.abs(detail["分數"]), 6)
        detail = detail.sort_values("絕對係數", ascending=False).reset_index(drop=True)
    elif method == METHOD_SFS:
        detail = detail.sort_values("是否選中", ascending=False, kind="stable").reset_index(drop=True)
    else:
        detail = detail.sort_values("分數", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return detail, label


def selected_feature_names(pipeline) -> list[str]:
    """取得最終被選中的轉換後特徵名稱。"""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    support = np.asarray(pipeline.named_steps["selector"].get_support(), dtype=bool)
    return [str(n) for n in names[support]]


def display_name(transformed_name: str) -> str:
    """移除 ColumnTransformer 的 num__/cat__ 前綴，方便閱讀。"""
    return re.sub(r"^(num|cat)__", "", transformed_name)
