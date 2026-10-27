"""圖表模組：所有圖表都回傳 matplotlib Figure，並可輸出為 PNG bytes 供下載。"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # 伺服器端（Streamlit Cloud）無 GUI，固定使用非互動後端

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

CHINESE_FONTS = [
    "Microsoft JhengHei",
    "PingFang TC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Heiti TC",
    "DejaVu Sans",
]

plt.rcParams["font.sans-serif"] = CHINESE_FONTS
plt.rcParams["axes.unicode_minus"] = False


def fig_to_png_bytes(fig) -> bytes:
    """將 Figure 轉為 PNG bytes，並關閉 Figure 以釋放記憶體。"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(corr) + 3), max(5, 0.6 * len(corr) + 2)))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            value = corr.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("數值變數相關係數熱圖")
    fig.tight_layout()
    return fig


def target_distribution(series: pd.Series, target_name: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna().to_numpy(dtype=float), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(target_name)
    ax.set_ylabel("次數")
    ax.set_title(f"目標變數「{target_name}」分布")
    fig.tight_layout()
    return fig


def feature_histograms(df: pd.DataFrame, columns: list[str], max_cols: int = 6):
    cols = columns[:max_cols]
    if not cols:
        return None
    n_cols = min(3, len(cols))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        ax.hist(df[col].dropna().to_numpy(dtype=float), bins=25, color="#55A868", edgecolor="white")
        ax.set_title(col, fontsize=10)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("數值特徵分布（最多顯示 6 個）")
    fig.tight_layout()
    return fig


def selection_bar_chart(detail: pd.DataFrame, score_label: str, method_name: str, top_n: int = 20):
    """特徵選擇分數長條圖（僅顯示前 top_n 個）。"""
    if detail is None or detail.empty or "分數" not in detail.columns:
        return None
    if detail["分數"].isna().all():
        return None
    view = detail.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(view) + 1)))
    colors = ["#C44E52" if s == "是" else "#8172B2" for s in view["是否選中"]]
    ax.barh(view["轉換後特徵"].astype(str), view["分數"].astype(float), color=colors)
    ax.set_xlabel(score_label)
    ax.set_title(f"{method_name}：特徵分數（紅色為被選中）")
    fig.tight_layout()
    return fig


COMPARISON_METRIC_NOTES = {
    "MAE": "越低越好",
    "MSE": "越低越好",
    "RMSE": "越低越好",
    "R²": "越高越好",
    "Adjusted R²": "越高越好",
}


def comparison_bar_chart(table: pd.DataFrame):
    """以 2x3 子圖比較五種特徵選擇方法的 MAE、MSE、RMSE、R² 與 Adjusted R²（測試集）。"""
    if table is None or table.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = table["特徵選擇方法"].astype(str)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for ax, metric in zip(axes.flat, COMPARISON_METRIC_NOTES):
        values = table[metric].astype(float).to_numpy()
        ax.barh(labels, values, color=colors[: len(values)])
        ax.set_title(f"{metric}（{COMPARISON_METRIC_NOTES[metric]}）")
        ax.set_xlabel(metric)
        ax.invert_yaxis()
    axes.flat[-1].set_visible(False)
    fig.suptitle("五種特徵選擇方法的測試集指標比較")
    fig.tight_layout()
    return fig


def actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#4C72B0")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
    ax.set_xlabel("實際值")
    ax.set_ylabel("預測值")
    ax.set_title("實際值 vs 預測值（測試集）")
    ax.legend()
    fig.tight_layout()
    return fig


def prediction_interval_plot(y_true, predictions: pd.DataFrame):
    """依預測值排序，畫出 95% 預測區間與實際值。"""
    order = np.argsort(predictions["預測值"].to_numpy(dtype=float))
    x = np.arange(len(order))
    pred = predictions["預測值"].to_numpy(dtype=float)[order]
    lower = predictions["95%預測區間下界"].to_numpy(dtype=float)[order]
    upper = predictions["95%預測區間上界"].to_numpy(dtype=float)[order]
    actual = np.asarray(y_true, dtype=float)[order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, lower, upper, color="#C7D9F0", alpha=0.8, label="95% 預測區間")
    ax.plot(x, pred, color="#4C72B0", label="預測值")
    ax.scatter(x, actual, s=18, color="#C44E52", label="實際值", zorder=3)
    ax.set_xlabel("測試集樣本（依預測值排序）")
    ax.set_ylabel("目標變數")
    ax.set_title("95% 預測區間與實際值")
    ax.legend()
    fig.tight_layout()
    return fig


def residuals_vs_fitted(fitted, residuals):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, residuals, alpha=0.7, color="#4C72B0")
    ax.axhline(0, color="gray", linestyle="--")
    try:
        smooth = lowess(residuals, fitted, frac=0.6, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color="#C44E52", label="LOWESS")
        ax.legend()
    except Exception:  # noqa: BLE001 - 平滑線失敗時仍保留散點圖
        pass
    ax.set_xlabel("擬合值")
    ax.set_ylabel("殘差")
    ax.set_title("殘差 vs 擬合值")
    fig.tight_layout()
    return fig


def scale_location(fitted, residuals):
    std_resid = residuals / (np.std(residuals) if np.std(residuals) > 0 else 1.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.7, color="#55A868")
    ax.set_xlabel("擬合值")
    ax.set_ylabel("√|標準化殘差|")
    ax.set_title("尺度-位置圖（Scale-Location）")
    fig.tight_layout()
    return fig


def residual_histogram(residuals):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=30, color="#8172B2", edgecolor="white")
    ax.set_xlabel("殘差")
    ax.set_ylabel("次數")
    ax.set_title("殘差直方圖")
    fig.tight_layout()
    return fig


def qq_plot(residuals):
    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.7, color="#4C72B0")
    ax.plot(osm, slope * np.asarray(osm) + intercept, color="#C44E52", linestyle="--")
    ax.set_xlabel("理論分位數")
    ax.set_ylabel("殘差分位數")
    ax.set_title("殘差 Q-Q Plot")
    fig.tight_layout()
    return fig


def vif_bar_chart(vif_table: pd.DataFrame, threshold: float):
    if vif_table is None or vif_table.empty or vif_table["VIF"].isna().all():
        return None
    view = vif_table.dropna(subset=["VIF"]).copy()
    finite = view["VIF"].replace([np.inf, -np.inf], np.nan)
    cap = float(np.nanmax(finite)) if finite.notna().any() else threshold
    view["VIF_plot"] = np.where(np.isfinite(view["VIF"].astype(float)), view["VIF"].astype(float), cap * 1.1)
    view = view.sort_values("VIF_plot")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(view) + 1)))
    ax.barh(view["特徵"].astype(str), view["VIF_plot"], color="#8172B2")
    ax.axvline(threshold, color="#C44E52", linestyle="--", label=f"門檻 {threshold:g}")
    ax.set_xlabel("VIF（無限大以長條末端表示）")
    ax.set_title("各特徵 VIF")
    ax.legend()
    fig.tight_layout()
    return fig
