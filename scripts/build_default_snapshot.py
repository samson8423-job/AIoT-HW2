"""建置預設分析（Problem 1：汽車價格預測）的預先計算快照。

用法（在專案根目錄執行）：
    python scripts/build_default_snapshot.py

會寫入 precomputed/default_analysis.json。資料檔、目標、特徵或分析設定變更後需重新執行。
設定值與 app.py 側邊欄的預設值保持一致；若兩者不同，快照會被略過並改為即時分析。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src import config  # noqa: E402
from src.analysis_runner import run_full_analysis  # noqa: E402
from src.chart_builders import data_understanding_charts, model_charts  # noqa: E402
from src.data_loader import load_local_csv  # noqa: E402
from src.data_preprocessing import clean_dataset, detect_id_like_columns  # noqa: E402
from src.feature_selection import METHOD_RFE  # noqa: E402
from src.precomputed import (  # noqa: E402
    SNAPSHOT_PATH,
    file_sha256,
    load_snapshot_file,
    save_snapshot,
    snapshot_result,
)


def main() -> None:
    started = time.perf_counter()
    target = config.CAR_TARGET
    data_path = config.CAR_LOCAL_CSV_PATH

    raw = load_local_csv(data_path)
    cleaned, _ = clean_dataset(raw, target)
    candidates = [c for c in cleaned.columns if c != target]
    id_cols = detect_id_like_columns(cleaned, target)
    features = [c for c in candidates if c not in id_cols]

    # 與 app.py 側邊欄預設值完全相同
    settings = {
        "test_size": float(config.DEFAULT_TEST_SIZE),
        "random_state": int(config.DEFAULT_RANDOM_STATE),
        "cv_folds": int(config.DEFAULT_CV_FOLDS),
        "k": int(config.DEFAULT_K),
        "direction": "forward",
        "final_method": METHOD_RFE,
        "alpha": config.ALPHA,
    }

    print(f"資料：{data_path}（{raw.shape[0]} 列 × {raw.shape[1]} 欄）")
    print(f"目標：{target}；特徵數：{len(features)}")
    result = run_full_analysis(cleaned, target, features, settings)
    charts_da = data_understanding_charts(cleaned, target)
    result["charts"] = model_charts(result)

    data_id = file_sha256(data_path)
    save_snapshot(
        SNAPSHOT_PATH,
        data_id=data_id,
        target=target,
        features=features,
        settings=settings,
        result=result,
        charts_da=charts_da,
    )
    print(f"已寫入：{SNAPSHOT_PATH}（{SNAPSHOT_PATH.stat().st_size / 1_048_576:.2f} MB）")

    # 驗證：讀回快照並重建模型，確認可被網站使用
    restored = snapshot_result(load_snapshot_file(SNAPSHOT_PATH), cleaned, target, features, settings, data_id)
    if restored is None:
        raise SystemExit("驗證失敗：快照無法還原，請檢查警告訊息。")
    metrics = restored["final_metrics"]
    print(f"驗證通過：R² = {metrics.get('R²')}，RMSE = {metrics.get('RMSE')}")
    print(f"總耗時 {time.perf_counter() - started:.1f} 秒")


if __name__ == "__main__":
    main()
