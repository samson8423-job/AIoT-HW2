"""資料清理與前處理模組。

此模組只做「與資料無關的通用清理」（去除重複列、移除全空欄、轉換數值字串等）。
需要從資料中學習的步驟（缺失值填補、標準化、One-Hot Encoding）
則封裝在 ColumnTransformer 中，於 Pipeline 內只使用訓練資料擬合，避免資料洩漏。
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# 測試集出現訓練集未見過的類別屬於預期情況，已依設計編入稀少類別，不需重複顯示警告
warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)

MISSING_TOKEN = "__缺失__"
_ID_NAME_RE = re.compile(config.ID_NAME_PATTERN, re.IGNORECASE)


def _to_numeric_if_mostly_numeric(series: pd.Series) -> pd.Series | None:
    """若非空值的大部分可轉為數值，回傳轉換後的序列，否則回傳 None。"""
    non_null = series.dropna()
    if non_null.empty:
        return None
    converted = pd.to_numeric(series, errors="coerce")
    ratio = converted.notna().sum() / len(non_null)
    if ratio >= config.TARGET_NUMERIC_RATIO:
        return converted
    return None


def clean_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """執行通用資料清理，回傳清理後資料與清理報告。

    Raises:
        ValueError: 目標欄位不存在、不是數值或無法轉換時。
    """
    if target not in df.columns:
        raise ValueError(f"目標欄位「{target}」不存在於資料中，請重新選擇。")

    report: dict = {
        "raw_rows": int(df.shape[0]),
        "raw_cols": int(df.shape[1]),
        "duplicates_removed": 0,
        "all_nan_dropped": [],
        "numeric_converted": [],
        "target_rows_dropped": 0,
        "constant_dropped": [],
        "final_rows": 0,
        "notes": [],
    }

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    # 1. 移除完全重複的列
    dup_count = int(work.duplicated().sum())
    work = work.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = dup_count

    # 2. 移除全部為缺失值的欄位
    all_nan = [c for c in work.columns if work[c].isna().all()]
    work = work.drop(columns=all_nan)
    report["all_nan_dropped"] = all_nan

    # 3. 將看似數值的文字欄位轉為數值（非空值至少 95% 可轉換）
    for col in work.columns:
        if col == target or is_numeric_dtype(work[col]) or is_bool_dtype(work[col]):
            continue
        if is_datetime64_any_dtype(work[col]):
            continue
        converted = _to_numeric_if_mostly_numeric(work[col])
        if converted is not None:
            work[col] = converted
            report["numeric_converted"].append(col)

    # 4. 檢查目標欄位
    if is_bool_dtype(work[target]):
        raise ValueError(f"目標欄位「{target}」為布林值，本平台只支援連續數值目標，請重新選擇。")
    if not is_numeric_dtype(work[target]):
        converted_target = _to_numeric_if_mostly_numeric(work[target])
        if converted_target is None:
            raise ValueError(
                f"目標欄位「{target}」是文字或類別資料，無法轉換為數值。"
                "本平台主要針對數值型連續目標變數，請重新選擇目標欄位。"
            )
        work[target] = converted_target
        report["notes"].append(f"目標欄位「{target}」已由文字轉換為數值。")

    # 5. 移除目標為缺失值的列（不可用缺失目標訓練）
    before = len(work)
    work = work.dropna(subset=[target]).reset_index(drop=True)
    report["target_rows_dropped"] = before - len(work)

    if work.empty:
        raise ValueError("清理後沒有可用資料列，請確認目標欄位是否大多為缺失值。")

    # 6. 移除常數欄位（只有單一值，對迴歸沒有資訊量）
    constant = [
        c for c in work.columns if c != target and work[c].nunique(dropna=True) <= 1
    ]
    work = work.drop(columns=constant)
    report["constant_dropped"] = constant

    # 7. 布林欄位轉為字串，以便作為類別特徵處理
    for col in work.columns:
        if col != target and is_bool_dtype(work[col]):
            work[col] = work[col].astype(object).where(work[col].isna(), work[col].astype(str))

    report["final_rows"] = int(work.shape[0])
    return work, report


def detect_id_like_columns(df: pd.DataFrame, target: str) -> list[str]:
    """偵測可能是 ID 或流水號的欄位（依名稱，或為唯一值等於列數的整數欄位）。"""
    n_rows = len(df)
    ids: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        if _ID_NAME_RE.search(str(col).strip()):
            ids.append(col)
            continue
        series = df[col]
        if is_integer_dtype(series) and n_rows >= 10 and series.nunique(dropna=False) == n_rows:
            ids.append(col)
    return ids


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], list[str]]:
    """將特徵分為數值型、類別型與無法使用（日期時間）三組。"""
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []
    for col in features:
        series = df[col]
        if is_datetime64_any_dtype(series):
            skipped.append(col)
        elif is_bool_dtype(series) or not is_numeric_dtype(series):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical, skipped


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """建立前處理器（尚未擬合）。

    數值型：中位數填補 + 標準化。
    類別型：固定值填補 + One-Hot Encoding（drop='first' 避免虛擬變數陷阱，
    限制最大類別數並將稀少類別歸為 infrequent，避免高基數欄位產生過多欄位）。
    """
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=config.MAX_CATEGORIES,
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("沒有可用的特徵欄位，請至少選擇一個數值或類別特徵。")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def profile_tables(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """產生資料理解用的摘要表：欄位型態、缺失值、描述性統計。"""
    dtype_df = pd.DataFrame(
        {
            "欄位": df.columns,
            "資料型態": [str(df[c].dtype) for c in df.columns],
            "角色": ["目標變數" if c == target else "候選特徵" for c in df.columns],
            "非空值數": [int(df[c].notna().sum()) for c in df.columns],
            "唯一值數": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    missing_df = pd.DataFrame(
        {
            "欄位": df.columns,
            "缺失值數": [int(df[c].isna().sum()) for c in df.columns],
            "缺失比例(%)": [round(float(df[c].isna().mean()) * 100, 2) for c in df.columns],
        }
    )
    numeric_df = df.select_dtypes(include="number")
    describe_df = numeric_df.describe().T.reset_index().rename(columns={"index": "欄位"})
    return {"dtypes": dtype_df, "missing": missing_df, "describe": describe_df}


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """數值欄位的 Pearson 相關係數矩陣（僅用於探索，不用於特徵選擇）。"""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method="pearson")
