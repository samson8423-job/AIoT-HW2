"""分析結果的磁碟快取：相同資料與設定時直接讀取，避免重複執行五種特徵選擇。"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import pandas as pd

CACHE_VERSION = 1  # 分析程式邏輯變更時請遞增，使舊快取失效
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def make_cache_key(df: pd.DataFrame, target: str, features: list[str], settings: dict) -> str:
    data_hash = int(pd.util.hash_pandas_object(df, index=False).sum())
    payload = {
        "version": CACHE_VERSION,
        "target": target,
        "features": list(features),
        "settings": settings,
        "data": data_hash,
    }
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_or_compute(key: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    path = CACHE_DIR / f"analysis_{key}.pkl"
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001 - 快取損毀時重新計算
            path.unlink(missing_ok=True)

    result = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(result, f)
    tmp.replace(path)
    return result
