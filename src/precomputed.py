"""預先建置的預設分析結果（JSON 快照）。

首次開啟網站時，若目前的資料與設定和快照一致，就直接讀取快照，不必重新執行
SFS／Lasso 等耗時分析。快照保存可序列化的結果與圖表（PNG 以 base64 編碼）；
最終模型（Pipeline 與 OLS）以快照記錄的特徵遮罩重新擬合，耗時約數秒，供預測頁使用。
重建後會比對快照中的預測值，不一致時回傳 None，由呼叫端改為重新分析。

快照由 scripts/build_default_snapshot.py 產生。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .analysis_cache import CACHE_VERSION
from .data_preprocessing import build_preprocessor
from .regression_model import fit_ols, selected_design_matrix

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "precomputed" / "default_analysis.json"

# 含有模型物件、無法序列化的鍵；載入時由 snapshot_result 重新建立
_NOT_SERIALIZED = ("final_pipeline", "ols")


class FixedSupportSelector(BaseEstimator, TransformerMixin):
    """以快照中記錄的特徵遮罩取出欄位。僅用於重建已完成的模型，本身不做任何選擇。"""

    def __init__(self, support=None):
        self.support = support

    def fit(self, X, y=None):
        # 以底線結尾的屬性表示已擬合（sklearn 據此判斷 Pipeline 是否可轉換）
        self.support_ = np.asarray(self.support, dtype=bool)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return self.support_


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_snapshot_key(data_id: str, target: str, features: list[str], settings: dict) -> str:
    """快照鍵：資料指紋、目標、特徵、設定與快取版本。任一項不同即不使用快照。"""
    payload = {
        "version": CACHE_VERSION,
        "data": data_id,
        "target": target,
        "features": list(features),
        "settings": settings,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 序列化：DataFrame / Series / ndarray / bytes 以標記型別編碼
# ---------------------------------------------------------------------------
def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, bytes):
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "columns": _encode(list(obj.columns)),
            "index": _encode(list(obj.index)),
            "data": [_encode(list(row)) for row in obj.itertuples(index=False, name=None)],
        }
    if isinstance(obj, pd.Series):
        return {
            "__type__": "Series",
            "name": _encode(obj.name),
            "index": _encode(list(obj.index)),
            "data": _encode(obj.tolist()),
        }
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": _encode(obj.tolist())}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    raise TypeError(f"無法序列化的型別：{type(obj)}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        kind = obj.get("__type__")
        if kind is None:
            return {k: _decode(v) for k, v in obj.items()}
        if kind == "bytes":
            return base64.b64decode(obj["data"])
        if kind == "ndarray":
            return np.asarray(_decode(obj["data"]))
        if kind == "DataFrame":
            rows = [_decode(row) for row in obj["data"]]
            return pd.DataFrame(rows, columns=_decode(obj["columns"]), index=pd.Index(_decode(obj["index"])))
        if kind == "Series":
            return pd.Series(_decode(obj["data"]), index=pd.Index(_decode(obj["index"])), name=_decode(obj["name"]))
        raise ValueError(f"未知的快照型別：{kind}")
    return obj


# ---------------------------------------------------------------------------
# 儲存與載入
# ---------------------------------------------------------------------------
def save_snapshot(
    path: Path,
    *,
    data_id: str,
    target: str,
    features: list[str],
    settings: dict,
    result: dict,
    charts_da: dict[str, bytes],
) -> None:
    """將分析結果寫入 JSON 快照。

    result["charts"] 應僅包含模型圖表；資料理解圖表以 charts_da 另存，載入時再合併。
    """
    serial = {k: v for k, v in result.items() if k not in _NOT_SERIALIZED}
    serial["method_results"] = [
        {k: v for k, v in entry.items() if k != "pipeline"} for entry in result["method_results"]
    ]
    doc = {
        "key": make_snapshot_key(data_id, target, features, settings),
        "data_id": data_id,
        "target": target,
        "features": list(features),
        "settings": settings,
        "charts_da": _encode(charts_da),
        "result": _encode(serial),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_snapshot_file(path: Path = SNAPSHOT_PATH) -> dict | None:
    """讀取並解碼快照檔；檔案不存在或損毀時回傳 None。"""
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {
            "key": doc["key"],
            "data_id": doc["data_id"],
            "target": doc["target"],
            "features": doc["features"],
            "settings": doc["settings"],
            "charts_da": _decode(doc["charts_da"]),
            "result": _decode(doc["result"]),
        }
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("預先建置的快照無法讀取，將改為即時分析：%s", exc)
        return None


# ---------------------------------------------------------------------------
# 模型重建
# ---------------------------------------------------------------------------
def _rebuild_final_model(result: dict, df: pd.DataFrame, target: str) -> tuple[Pipeline, Any]:
    settings = result["settings"]
    features = list(result["features_input"])
    X = df[features].copy()
    y = df[target].astype(float)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=float(settings["test_size"]), random_state=int(settings["random_state"])
    )

    final_selected = [str(n) for n in result["final_selected"]]
    preprocessor = build_preprocessor(list(result["numeric_cols"]), list(result["categorical_cols"]))
    preprocessor.fit(X_train, y_train)
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    chosen = set(final_selected)
    support = np.array([n in chosen for n in names], dtype=bool)
    if int(support.sum()) != len(final_selected):
        raise ValueError("重建的特徵數量與快照不一致。")

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("selector", FixedSupportSelector(support)),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    stored = result["predictions"]["預測值"].to_numpy(dtype=float)
    if not np.allclose(pipeline.predict(X_test), stored, rtol=1e-6, atol=1e-6):
        raise ValueError("重建模型的預測值與快照不一致。")

    ols = fit_ols(selected_design_matrix(pipeline, X_train), y_train)
    return pipeline, ols


def snapshot_result(
    snapshot: dict | None,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    settings: dict,
    data_id: str,
) -> dict | None:
    """若快照與目前輸入相符，回傳還原後的完整分析結果；否則回傳 None。"""
    if snapshot is None or not data_id:
        return None
    if snapshot["key"] != make_snapshot_key(data_id, target, features, settings):
        return None
    result = dict(snapshot["result"])
    try:
        pipeline, ols = _rebuild_final_model(result, df, target)
    except Exception as exc:  # 任何重建失敗都改為即時分析，不讓頁面出錯
        logger.warning("快照模型重建失敗，將改為即時分析：%s", exc)
        return None
    result["final_pipeline"] = pipeline
    result["ols"] = ols
    return result
