"""圖表組合：資料理解與模型結果的 PNG 圖表。

網站即時分析與預先建置的預設結果（scripts/build_default_snapshot.py）共用本模組，
確保兩者產生的圖表完全一致。
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config
from . import visualization as viz
from .data_preprocessing import correlation_matrix
from .feature_selection import METHOD_SFS, METHOD_SHORT
from .visualization import fig_to_png_bytes


def data_understanding_charts(cleaned: pd.DataFrame, target: str) -> dict[str, bytes]:
    charts: dict[str, bytes] = {}
    corr = correlation_matrix(cleaned)
    if not corr.empty:
        charts["相關係數熱圖"] = fig_to_png_bytes(viz.correlation_heatmap(corr))
    charts["目標變數分布"] = fig_to_png_bytes(viz.target_distribution(cleaned[target], target))
    numeric_feats = [c for c in cleaned.columns if c != target and is_numeric_dtype(cleaned[c])]
    fig = viz.feature_histograms(cleaned, numeric_feats)
    if fig is not None:
        charts["數值特徵分布"] = fig_to_png_bytes(fig)
    return charts


def model_charts(result: dict) -> dict[str, bytes]:
    charts: dict[str, bytes] = {}
    for entry in result["method_results"]:
        if entry["detail"] is None or entry["method"] == METHOD_SFS:
            continue
        fig = viz.selection_bar_chart(entry["detail"], entry["score_label"], entry["method"])
        if fig is not None:
            charts[f"特徵分數_{METHOD_SHORT[entry['method']]}"] = fig_to_png_bytes(fig)

    fig = viz.comparison_bar_chart(result["comparison_table"])
    if fig is not None:
        charts["五種方法指標比較"] = fig_to_png_bytes(fig)

    pred = result["predictions"]
    charts["實際值與預測值"] = fig_to_png_bytes(
        viz.actual_vs_predicted(pred["實際值"].to_numpy(), pred["預測值"].to_numpy())
    )
    charts["95%預測區間"] = fig_to_png_bytes(viz.prediction_interval_plot(pred["實際值"].to_numpy(), pred))

    a = result["assumptions"]
    charts["殘差對擬合值"] = fig_to_png_bytes(viz.residuals_vs_fitted(a["fitted"], a["residuals"]))
    charts["尺度位置圖"] = fig_to_png_bytes(viz.scale_location(a["fitted"], a["residuals"]))
    charts["殘差直方圖"] = fig_to_png_bytes(viz.residual_histogram(a["residuals"]))
    charts["殘差QQ圖"] = fig_to_png_bytes(viz.qq_plot(a["residuals"]))
    fig = viz.vif_bar_chart(a["vif_table"], config.VIF_THRESHOLD)
    if fig is not None:
        charts["VIF長條圖"] = fig_to_png_bytes(fig)
    return charts
