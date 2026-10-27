"""多元線性迴歸分析平台：Streamlit 入口（app.py）。

執行方式：python -m streamlit run app.py
所有分析邏輯位於 src/ 目錄，本檔案只負責介面與流程編排。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from src import ai_content, config
from src import visualization as viz
from src.analysis_cache import load_or_compute, make_cache_key
from src.analysis_runner import run_full_analysis
from src.data_loader import (
    DataLoadError,
    load_csv_from_url,
    load_local_csv,
    load_startup_50,
    load_uploaded_csv,
)
from src.data_preprocessing import (
    clean_dataset,
    correlation_matrix,
    detect_id_like_columns,
    profile_tables,
)
from src.feature_selection import (
    METHOD_DESCRIPTIONS,
    METHOD_KBEST,
    METHOD_LASSO,
    METHOD_PEARSON,
    METHOD_RFE,
    METHOD_SFS,
    METHODS,
    display_name,
)
from src.regression_model import prediction_interval, selected_design_matrix
from src.report_generator import (
    build_export_tables,
    build_markdown_report,
    df_to_markdown,
    to_csv_bytes,
)
from src.visualization import fig_to_png_bytes

st.set_page_config(page_title="多元線性迴歸分析平台", layout="wide")

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
P1 = "Problem 1：汽車價格預測（Car Price Prediction）"
P2 = "Problem 2：自訂資料集迴歸分析（Custom Dataset Regression）"

SRC_LOCAL = "本機 data/data.csv"
SRC_UPLOAD = "上傳 CSV 檔案"
SRC_URL = "公開 CSV 網址"
SRC_STARTUP = "Startup 50 範例資料集"

PROBLEM_SOURCES = {
    P1: [SRC_LOCAL, SRC_UPLOAD, SRC_URL],
    P2: [SRC_URL, SRC_UPLOAD, SRC_STARTUP],
}

METHOD_SHORT = {
    METHOD_PEARSON: "Pearson",
    METHOD_KBEST: "SelectKBest",
    METHOD_RFE: "RFE",
    METHOD_SFS: "SFS",
    METHOD_LASSO: "Lasso",
}

TAB_NAMES = [
    "專案介紹",
    "商業理解",
    "資料理解",
    "資料準備",
    "建模與特徵選擇",
    "評估與預測區間",
    "迴歸假設檢驗",
    "部署與預測",
    "AI 輔助與研究摘要",
    "報告與下載",
]


# ---------------------------------------------------------------------------
# 資料載入（快取；例外不會被快取，因此錯誤可重新嘗試）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="正在讀取 Startup 50 範例資料…", ttl=3600)
def cached_startup() -> pd.DataFrame:
    return load_startup_50()


@st.cache_data(show_spinner="正在下載公開 CSV…", ttl=3600)
def cached_url(url: str) -> pd.DataFrame:
    return load_csv_from_url(url)


@st.cache_data(show_spinner="正在讀取本機 CSV…")
def cached_local(mtime: float) -> pd.DataFrame:
    # mtime 只作為快取鍵，檔案更新後會重新讀取
    return load_local_csv(config.CAR_LOCAL_CSV_PATH)


@st.cache_data(show_spinner="正在解析上傳的 CSV…")
def cached_upload(content: bytes, file_name: str) -> pd.DataFrame:
    return load_uploaded_csv(content, file_name)


@st.cache_data(show_spinner="正在執行分析，完成後會快取以加速下次開啟…", max_entries=8)
def cached_analysis(df: pd.DataFrame, target: str, features: tuple, settings: dict) -> dict:
    def compute() -> dict:
        res = run_full_analysis(df, target, list(features), settings)
        res["charts"] = {**data_understanding_charts(df, target), **model_charts(res)}
        return res

    return load_or_compute(make_cache_key(df, target, list(features), settings), compute)


def local_csv_mtime() -> float:
    try:
        return config.CAR_LOCAL_CSV_PATH.stat().st_mtime
    except OSError:
        return 0.0


def load_raw(source: str, url: str, upload) -> tuple[pd.DataFrame, str]:
    """依來源讀取原始資料，回傳 (資料, 來源說明)。失敗時拋出 DataLoadError。"""
    if source == SRC_STARTUP:
        return cached_startup(), f"Startup 50 範例資料集（{config.STARTUP_URL}）"
    if source == SRC_LOCAL:
        return cached_local(local_csv_mtime()), "本機檔案 data/data.csv（使用者自行提供）"
    if source == SRC_URL:
        clean_url = url.strip()
        if not clean_url:
            raise DataLoadError("請在側邊欄輸入公開 CSV 網址。")
        return cached_url(clean_url), f"公開 CSV 網址：{clean_url}"
    if source == SRC_UPLOAD:
        if upload is None:
            raise DataLoadError("請在側邊欄上傳 CSV 檔案。")
        return cached_upload(upload.getvalue(), upload.name), f"上傳檔案：{upload.name}"
    raise DataLoadError("未知的資料來源。")


def default_target_index(problem: str, columns: list[str], raw: pd.DataFrame) -> int:
    preferred = config.CAR_TARGET if problem == P1 else config.STARTUP_TARGET
    if preferred in columns:
        return columns.index(preferred)
    numeric = [c for c in columns if is_numeric_dtype(raw[c])]
    if numeric:
        return columns.index(numeric[-1])
    return 0


# ---------------------------------------------------------------------------
# 圖表
# ---------------------------------------------------------------------------
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


def show_chart(charts: dict[str, bytes], name: str) -> None:
    if name in charts:
        st.image(charts[name], caption=name)


# ---------------------------------------------------------------------------
# 分頁內容
# ---------------------------------------------------------------------------
def render_intro(problem: str) -> None:
    st.header("專案介紹")
    st.write(
        "本平台以多元線性迴歸（Multiple Linear Regression）進行預測分析，流程遵循 CRISP-DM 六階段，"
        "比較五種特徵選擇方法，並提供迴歸假設檢驗、95% 預測區間與結果下載。"
    )
    if problem == P1:
        st.subheader("Problem 1：汽車價格預測")
        st.write(
            "資料集為 Kaggle 公開資料集 **Car Features and MSRP**（識別名稱：CooperUnion/cardataset），"
            "預測目標為 MSRP（建議售價）。"
        )
        st.markdown(f"官方資料集頁面：[{config.CAR_KAGGLE_PAGE}]({config.CAR_KAGGLE_PAGE})")
        st.info(
            "Kaggle 資料集需要登入才能下載。請先自行下載 CSV，並選擇下列任一方式：\n\n"
            "1. 將檔案另存為 data/data.csv（放在專案的 data 資料夾）。\n"
            "2. 在側邊欄選擇「上傳 CSV 檔案」。\n"
            "3. 在側邊欄選擇「公開 CSV 網址」，輸入可公開讀取的 CSV 連結。\n\n"
            "程式不會在無法取得資料時產生假的分析結果。"
        )
    else:
        st.subheader("Problem 2：自訂資料集迴歸分析")
        st.write(
            "可選擇公開 CSV 網址、上傳本機 CSV，或使用內建的 Startup 50 範例資料集（預設目標變數為 Profit）。"
            "資料載入後，可自由選擇目標變數與輸入特徵。"
        )
        st.caption(
            f"注意：Startup 50 原始只有 4 個輸入欄位，轉換後約 5 個特徵，少於本作業要求的 "
            f"{config.MIN_EFFECTIVE_FEATURES} 個有效特徵，平台會明確提示此限制。"
        )
    st.subheader("使用方式")
    st.markdown(
        "1. 在頁面上方選擇 Problem。\n"
        "2. 在側邊欄選擇資料來源、目標變數與輸入特徵。\n"
        "3. 調整測試集比例、Random State、交叉驗證與 K 值。\n"
        "4. 按下「執行分析」，依序檢視各分頁結果與圖表。\n"
        "5. 在「報告與下載」分頁下載 CSV、Markdown 報告與 PNG 圖表。"
    )


def render_business(problem: str, target: str | None, features: list[str]) -> None:
    st.header("商業理解（Business Understanding）")
    if problem == P1:
        question = "能否利用汽車的基本規格與性能資訊，建立多元線性迴歸模型，預測汽車的建議售價？"
        st.write(f"**研究問題**：{question}")
        st.write(
            "**研究目的**：協助理解哪些規格（如引擎馬力、年份、油耗、車輛尺寸、品牌）與建議售價相關，"
            "並評估線性模型的預測能力。"
        )
        st.write("**可能影響價格的特徵**：品牌、車型、年份、引擎馬力、汽缸數、變速箱、驅動方式、油耗與車身尺寸。")
    else:
        if target:
            st.write(
                f"**研究問題**：能否利用所選的 {len(features)} 個輸入特徵，建立多元線性迴歸模型，預測「{target}」？"
            )
        else:
            st.write("**研究問題**：請先載入資料並選擇目標變數。")
        st.write("**研究目的**：檢驗所選特徵與目標變數之間是否存在可用於線性預測的關係。")
    st.write("**實際應用價值**：可作為初步的基準模型，輔助估計數值型結果，並說明影響因素的方向與強度。")
    st.write(
        "**研究限制**：線性模型只能捕捉線性關係；相關性不代表因果關係；"
        "結果只適用於與訓練資料特性相近的情境。"
    )


def render_data_understanding(raw, cleaned, target, source_desc, charts_da) -> None:
    st.header("資料理解（Data Understanding）")
    if raw is None:
        st.info("尚未載入資料，請於側邊欄設定資料來源。")
        return
    st.write(f"**資料來源**：{source_desc}")
    duplicates = int(raw.duplicated().sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("原始資料筆數", f"{raw.shape[0]:,}")
    col2.metric("原始欄位數", f"{raw.shape[1]}")
    col3.metric("重複資料筆數", f"{duplicates:,}")
    col4.metric("目標變數", target or "未選擇")
    if raw.shape[1] != config.CAR_EXPECTED_RAW_COLUMNS and target == config.CAR_TARGET:
        st.caption(
            f"注意：原始欄位數為 {raw.shape[1]}，與預期的 {config.CAR_EXPECTED_RAW_COLUMNS} 不同，以實際讀取結果為準。"
        )

    st.subheader("原始資料前五筆")
    st.dataframe(raw.head(5), hide_index=True)

    if cleaned is None:
        st.warning("目標變數無法用於迴歸分析，請於側邊欄重新選擇。")
        return
    tables = profile_tables(cleaned, target)
    st.subheader("欄位型態")
    st.dataframe(tables["dtypes"], hide_index=True)
    st.subheader("缺失值統計")
    st.dataframe(tables["missing"], hide_index=True)
    st.subheader("描述性統計（數值欄位）")
    st.dataframe(tables["describe"], hide_index=True)
    st.subheader("數值欄位相關係數矩陣")
    corr = correlation_matrix(cleaned)
    if corr.empty:
        st.info("數值欄位不足兩個，無法計算相關係數矩陣。")
    else:
        st.dataframe(corr.round(4))
    st.caption("相關係數只描述線性關聯強度，不代表因果關係。")
    for name in ["目標變數分布", "相關係數熱圖", "數值特徵分布"]:
        show_chart(charts_da, name)


def render_preparation(cleaned, clean_report, target, features, test_size, random_state) -> None:
    st.header("資料準備（Data Preparation）")
    if clean_report is None:
        st.info("尚無可用的清理結果。")
        return
    st.subheader("資料清理報告")
    st.write(f"- 原始資料：{clean_report['raw_rows']:,} 列，{clean_report['raw_cols']} 欄")
    st.write(f"- 移除重複資料：{clean_report['duplicates_removed']:,} 列")
    st.write(f"- 移除全部為缺失值的欄位：{', '.join(clean_report['all_nan_dropped']) or '無'}")
    st.write(f"- 由文字轉為數值的欄位：{', '.join(clean_report['numeric_converted']) or '無'}")
    st.write(f"- 移除目標變數為缺失值的列：{clean_report['target_rows_dropped']:,} 列")
    st.write(f"- 移除常數欄位（只有單一值）：{', '.join(clean_report['constant_dropped']) or '無'}")
    st.write(f"- 清理後資料：{clean_report['final_rows']:,} 列")
    for note in clean_report["notes"]:
        st.info(note)

    st.subheader("前處理規則")
    st.markdown(
        "- **數值型特徵**：中位數填補缺失值，再以 StandardScaler 標準化。\n"
        "- **類別型特徵**：以固定類別「__缺失__」填補，再進行 One-Hot Encoding（drop='first' 避免虛擬變數陷阱）；"
        f"最多保留 {config.MAX_CATEGORIES} 個類別，其餘歸為稀少類別。\n"
        "- **Pipeline**：前處理、特徵選擇與 LinearRegression 皆包在 sklearn Pipeline 中，只在訓練集上擬合。"
    )
    st.subheader("訓練與測試切分")
    st.write(
        f"測試集比例 {test_size:.0%}，Random State = {random_state}，"
        f"訓練集約 {1 - test_size:.0%}。特徵選擇只使用訓練資料，測試集只用於最終評估。"
    )
    if target and features:
        st.write(f"目標變數：{target}；候選輸入特徵：{', '.join(features)}")


def render_selection(result) -> None:
    st.header("建模與特徵選擇（Modeling）")
    if result is None:
        st.info("請於側邊欄按下「執行分析」後顯示結果。")
        return
    if result["effective_feature_warning"]:
        st.warning(
            f"轉換後只有 {result['n_transformed_features']} 個有效特徵，少於本作業要求的 "
            f"{config.MIN_EFFECTIVE_FEATURES} 個，結果不一定符合作業的特徵數量要求。"
        )
    for entry in result["method_results"]:
        with st.expander(entry["method"], expanded=False):
            st.write(entry["description"])
            if entry["error"]:
                st.error(f"此方法執行失敗：{entry['error']}")
                continue
            st.write(f"**選擇數量**：{entry['n_selected']}")
            st.write("**選中的特徵**：" + ("、".join(display_name(f) for f in entry["selected"]) or "無"))
            for note in entry["notes"]:
                st.info(note)
            if entry["detail"] is not None:
                st.dataframe(entry["detail"], hide_index=True)
            if entry["method"] == METHOD_SFS:
                st.caption(
                    "SFS 以交叉驗證 RMSE 逐步加入（前向）或移除（後向）特徵，"
                    "每一步都只使用訓練資料，不會看到測試集。"
                )
            show_chart(result["charts"], f"特徵分數_{METHOD_SHORT[entry['method']]}")

    st.subheader("五種方法比較（測試集指標）")
    table = result["comparison_table"]
    if table.empty:
        st.error("所有方法都無法完成分析，請檢查上方錯誤訊息。")
        return
    st.dataframe(table, hide_index=True)
    st.caption("表格可點擊欄位標題排序；預設依測試集 RMSE 由低至高排列。")
    best = table.iloc[0]
    st.write(
        f"依測試集 RMSE 排序，表現最佳的是「{best['特徵選擇方法']}」（RMSE={best['RMSE']:.4f}，"
        f"R²={best['R²']:.4f}）。單一測試集的排名可能受抽樣影響，建議同時參考 R² 與方法的可解釋性。"
    )
    show_chart(result["charts"], "五種方法指標比較")


def render_evaluation(result) -> None:
    st.header("評估與預測區間（Evaluation）")
    if result is None:
        st.info("請於側邊欄按下「執行分析」後顯示結果。")
        return
    m = result["final_metrics"]
    st.write(f"最終模型：**{result['final_method']}**（已選 {m['p_predictors']} 個轉換後特徵）")
    cols = st.columns(5)
    cols[0].metric("MAE", f"{m['MAE']:.4f}")
    cols[1].metric("MSE", f"{m['MSE']:.4f}")
    cols[2].metric("RMSE", f"{m['RMSE']:.4f}")
    cols[3].metric("R²", f"{m['R²']:.4f}")
    adj = m["Adjusted R²"]
    cols[4].metric("Adjusted R²", "N/A" if np.isnan(adj) else f"{adj:.4f}")

    st.markdown(
        "- **MAE / MSE / RMSE**：預測誤差的大小，RMSE 與目標變數單位相同，越低越好。\n"
        "- **R²**：測試集上模型解釋變異的比例，可能為負值，不應強制限制在 0～1。\n"
        "- **Adjusted R²**：公式 Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)，"
        "n 為測試集樣本數，p 為最終模型實際使用的轉換後特徵數（不含截距）。"
        "它會依特徵數量懲罰 R²，避免只因加入變數而使 R² 變高。"
    )
    st.caption(m["Adjusted R² 說明"])
    st.write(
        f"測試集樣本數 n = {m['n_test']}，有效預測變數數 p = {m['p_predictors']}。"
        "上述指標皆由測試集的實際預測結果計算，不是訓練集上的 R²。"
    )

    st.subheader("實際值與預測值")
    show_chart(result["charts"], "實際值與預測值")

    st.subheader("95% 預測區間")
    st.markdown(
        "預測區間描述**個別新觀測值**的不確定性，通常比平均反應的信賴區間更寬，"
        "且不是把預測值上下加減固定百分比。區間由 statsmodels OLS 的 `get_prediction().summary_frame()` 計算。"
    )
    show_chart(result["charts"], "95%預測區間")
    st.caption(
        "多個特徵時不能用單一維度的迴歸線代表完整模型，因此圖中以測試集樣本依預測值排序呈現。"
    )

    st.subheader("OLS 係數表（訓練集）")
    st.dataframe(result["ols_coefficients"], hide_index=True)
    st.write(
        f"交叉檢查：sklearn Pipeline 與 statsmodels OLS 在測試集的預測值最大絕對差異為 "
        f"{result['sklearn_vs_ols_max_diff']:.2e}。"
    )

    st.subheader("測試集預測結果")
    st.dataframe(result["predictions"], hide_index=True)


def render_assumptions(result) -> None:
    st.header("迴歸假設檢驗（Assumptions）")
    if result is None:
        st.info("請於側邊欄按下「執行分析」後顯示結果。")
        return
    a = result["assumptions"]
    st.write(
        "以下檢定皆以最終模型在訓練集上的 OLS 殘差進行診斷，顯著水準 α = "
        f"{a['alpha']:.2f}。統計檢定的 p-value 不是「假設成立的機率」；"
        "未拒絕虛無假設只代表「未發現足夠證據拒絕」，不代表假設已被證明成立。"
    )
    st.subheader("假設檢驗總表")
    st.dataframe(a["hypothesis_table"], hide_index=True)

    st.subheader("1～2. 線性關係與誤差獨立性")
    show_chart(result["charts"], "殘差對擬合值")
    st.write(
        "殘差若圍繞零值隨機分布，代表線性假設較合理；若呈現彎曲或系統性趨勢，則可能存在非線性。"
        "Rainbow 檢定只是輔助證據，不能取代殘差圖判讀。"
    )
    st.write(
        "Durbin-Watson 統計量接近 2 代表較少的一階自相關證據。它主要針對資料順序或時間相關的殘差，"
        "一般橫斷面資料不能僅憑此檢定宣稱誤差完全獨立。"
    )

    st.subheader("3. 誤差等變異性")
    show_chart(result["charts"], "尺度位置圖")
    bp = a["homoscedasticity"]
    if "error" in bp:
        st.error(bp["error"])
    else:
        st.write(
            f"Breusch-Pagan：LM 統計量 = {bp['LM 統計量']:.4f}，p-value = {bp['LM p-value']:.4g}；"
            f"F 統計量 = {bp['F 統計量']:.4f}，p-value = {bp['F p-value']:.4g}（α = {a['alpha']:.2f}）。"
        )
        if bp["LM p-value"] < a["alpha"]:
            st.write("結果：有證據顯示殘差變異數可能不固定。")
        else:
            st.write("結果：未發現足夠證據拒絕等變異性假設。")
        st.caption("單一檢定不能完全排除或證明所有異質變異問題，應搭配殘差圖一起判讀。")

    st.subheader("4. 殘差常態性")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(result["charts"], "殘差直方圖")
    with c2:
        show_chart(result["charts"], "殘差QQ圖")
    sw = a["normality"]
    if "error" in sw:
        st.error(sw["error"])
    else:
        st.write(f"Shapiro-Wilk：W = {sw['統計量']:.4f}，p-value = {sw['p-value']:.4g}（α = {a['alpha']:.2f}）。")
        if sw["note"]:
            st.caption(sw["note"])
        st.caption("樣本量大時 Shapiro-Wilk 對輕微偏離很敏感，請同時觀察 Q-Q Plot。")

    st.subheader("5. 多重共線性（VIF）")
    st.dataframe(a["vif_table"], hide_index=True)
    for note in a["vif_notes"]:
        st.warning(note)
    st.caption(
        f"VIF 大於 {config.VIF_THRESHOLD:g} 是常見的經驗提示門檻，並非絕對標準。"
        "類別變數經 One-Hot Encoding 後若出現完全共線性（例如冗餘虛擬變數），VIF 會成為無限大，"
        "此時需檢視特徵組合。"
    )
    show_chart(result["charts"], "VIF長條圖")


def predict_single(result: dict, values: dict) -> tuple[float, pd.Series]:
    """以與訓練相同的 Pipeline 與 OLS 計算單筆預測值與 95% 預測區間。"""
    row = pd.DataFrame([values])
    pipe = result["final_pipeline"]
    pred = float(pipe.predict(row)[0])
    design = selected_design_matrix(pipe, row)
    pi = prediction_interval(result["ols"], design, alpha=result["alpha"]).iloc[0]
    return pred, pi


def render_deployment(result, cleaned, target) -> None:
    st.header("部署與預測（Deployment）")
    st.markdown(
        "模型可用於快速估計與訓練資料相近的新資料，例如輸入規格後取得預測值與 95% 預測區間。\n\n"
        "**限制**：預測值不代表實際成交價格；模型只適用於與訓練資料特性相近的資料；"
        "線性模型無法自動捕捉複雜的非線性關係。"
    )
    if result is None or cleaned is None:
        st.info("請先執行分析，再使用單筆預測功能。")
        return

    inputs = result["numeric_cols"] + result["categorical_cols"]
    st.subheader(f"單筆預測（目標：{target}）")
    st.caption("輸入值會經過與訓練時相同的前處理與特徵選擇流程。")
    values: dict = {}
    with st.form("single_prediction_form"):
        for idx, col in enumerate(inputs):
            if col in result["numeric_cols"]:
                median = float(cleaned[col].median())
                values[col] = st.number_input(col, value=median, key=f"pred_num_{idx}_{col}")
            else:
                options = sorted(cleaned[col].dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                values[col] = st.selectbox(col, options, key=f"pred_cat_{idx}_{col}")
        submitted = st.form_submit_button("預測")

    if submitted:
        try:
            pred, pi = predict_single(result, values)
            c1, c2, c3 = st.columns(3)
            c1.metric("預測值", f"{pred:,.4f}")
            c2.metric("95% 預測區間下界", f"{pi['obs_ci_lower']:,.4f}")
            c3.metric("95% 預測區間上界", f"{pi['obs_ci_upper']:,.4f}")
            st.caption(
                "此區間為個別新觀測值的 95% 預測區間（statsmodels obs_ci），並非信賴區間，也不代表實際成交價格。"
            )
        except Exception as exc:  # noqa: BLE001 - 單筆預測失敗時不讓頁面崩潰
            st.error(f"單筆預測失敗：{type(exc).__name__}: {exc}")


def render_ai(problem: str) -> None:
    st.header("AI 輔助與研究摘要")
    st.subheader("GPT 輔助內容")
    for item in ai_content.GPT_ASSIST_ITEMS:
        st.markdown(f"- {item}")
    st.info(ai_content.GPT_DISCLAIMER)

    st.subheader(ai_content.NOTEBOOKLM_DRAFT_LABEL)
    st.warning(
        "以下為草稿，尚未經過 NotebookLM 對實際參考來源的研究與驗證，請勿視為已完成的文獻分析。"
    )
    st.write(ai_content.NOTEBOOKLM_DRAFT_TEXT)
    st.caption(f"草稿字數（不含空白）：{len(ai_content.NOTEBOOKLM_DRAFT_TEXT.replace(' ', ''))}")

    st.subheader("研究資料欄位（請於 NotebookLM 研究後填寫）")
    st.text_input("研究主題", value="利用多元線性迴歸進行汽車價格預測、特徵選擇與模型診斷", key="nb_topic")
    st.text_input("參考來源名稱", key="nb_source_name")
    st.text_input("來源連結", key="nb_source_url")
    st.text_area("NotebookLM 產生的摘要", key="nb_generated", height=160)
    st.text_area("學生修訂後的摘要", key="nb_revised", height=160)
    st.download_button(
        "下載研究紀錄範本（Markdown）",
        data=ai_content.RESEARCH_LOG_TEMPLATE.encode("utf-8"),
        file_name="notebooklm_research_log_template.md",
        mime="text/markdown",
        key="dl_research_template",
    )
    st.caption("欄位內容僅保存在目前的瀏覽器工作階段中，不會寫入伺服器。")


def render_downloads(result, problem: str, source_desc: str, charts: dict[str, bytes]) -> None:
    st.header("報告與下載")
    if result is None:
        st.info("請先執行分析，才能下載結果。")
        return
    st.subheader("CSV 結果")
    for idx, (name, df) in enumerate(build_export_tables(result).items()):
        st.download_button(
            f"下載 {name}",
            data=to_csv_bytes(df),
            file_name=name,
            mime="text/csv",
            key=f"dl_csv_{idx}",
        )

    st.subheader("Markdown 完整報告")
    report = build_markdown_report(result, problem_label=problem, source_description=source_desc)
    st.download_button(
        "下載完整分析報告（Markdown）",
        data=report.encode("utf-8"),
        file_name="分析報告.md",
        mime="text/markdown",
        key="dl_report_md",
    )

    st.subheader("圖表（PNG）")
    for idx, (name, png) in enumerate(charts.items()):
        st.download_button(
            f"下載 {name}.png",
            data=png,
            file_name=f"{name}.png",
            mime="image/png",
            key=f"dl_png_{idx}",
        )
    st.caption("所有下載檔皆由目前的分析結果產生；伺服器不會永久保存分析結果或上傳的資料。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
st.title("多元線性迴歸分析平台")
problem = st.selectbox("選擇 Problem", [P1, P2], index=0, key="problem")
pk = "p1" if problem == P1 else "p2"

with st.sidebar:
    st.header("分析設定")
    st.subheader("1. 資料來源")
    source = st.radio("資料來源", PROBLEM_SOURCES[problem], key=f"source_{pk}")
    url = ""
    upload = None
    if source == SRC_URL:
        url = st.text_input("公開 CSV 網址", key=f"url_{pk}", placeholder="https://.../data.csv")
    if source == SRC_UPLOAD:
        upload = st.file_uploader("上傳 CSV", type=["csv"], key=f"upload_{pk}")

raw: pd.DataFrame | None = None
source_desc = ""
load_error: str | None = None
try:
    raw, source_desc = load_raw(source, url, upload)
except DataLoadError as exc:
    load_error = str(exc)
except Exception as exc:  # noqa: BLE001 - 網路或解析錯誤不應讓網站崩潰
    load_error = f"資料讀取失敗：{type(exc).__name__}: {exc}"

if load_error:
    st.error(load_error)
    if problem == P1 and source == SRC_LOCAL:
        st.info("若尚未準備 data/data.csv，請改用「上傳 CSV 檔案」或「公開 CSV 網址」。")

cleaned: pd.DataFrame | None = None
clean_report: dict | None = None
target: str | None = None
features: list[str] = []
settings: dict | None = None
data_key = ""

with st.sidebar:
    if raw is not None:
        data_key = f"{pk}_{abs(hash((source, url, getattr(upload, 'name', ''), len(raw))))}"
        columns = [str(c) for c in raw.columns]
        st.subheader("2. 目標變數與特徵")
        target = st.selectbox(
            "目標變數",
            columns,
            index=default_target_index(problem, columns, raw),
            key=f"target_{data_key}",
        )
        try:
            cleaned, clean_report = clean_dataset(raw, target)
        except ValueError as exc:
            st.error(str(exc))
            cleaned = None

        if cleaned is not None:
            candidates = [c for c in cleaned.columns if c != target]
            id_cols = detect_id_like_columns(cleaned, target)
            default_features = [c for c in candidates if c not in id_cols]
            features = st.multiselect(
                "輸入特徵",
                candidates,
                default=default_features,
                key=f"features_{data_key}_{target}",
            )
            if id_cols:
                st.caption(f"已偵測為 ID 類欄位（預設排除）：{', '.join(id_cols)}")
            if len(features) < config.MIN_EFFECTIVE_FEATURES:
                st.warning(
                    f"目前只選擇 {len(features)} 個輸入特徵，少於本作業要求的 "
                    f"{config.MIN_EFFECTIVE_FEATURES} 個有效特徵。One-Hot 後的實際數量可能更少或更多，"
                    "分析結果請依此謹慎解讀。"
                )

            st.subheader("3. 模型設定")
            test_size = st.slider("測試集比例", 0.10, 0.40, float(config.DEFAULT_TEST_SIZE), 0.05, key=f"ts_{pk}")
            random_state = st.number_input(
                "Random State", min_value=0, max_value=1_000_000, value=config.DEFAULT_RANDOM_STATE, step=1, key=f"rs_{pk}"
            )
            cv_folds = st.slider("交叉驗證折數（SFS／Lasso）", 3, 10, config.DEFAULT_CV_FOLDS, key=f"cv_{pk}")
            k_value = st.number_input(
                "K（選擇特徵數量）", min_value=1, max_value=500, value=config.DEFAULT_K, step=1, key=f"k_{pk}"
            )
            direction = st.radio(
                "SFS 方向",
                ["forward", "backward"],
                format_func=lambda d: "前向選擇（forward）" if d == "forward" else "後向選擇（backward）",
                key=f"dir_{pk}",
            )
            final_method = st.selectbox("最終模型採用的特徵選擇方法", METHODS, index=1, key=f"final_{pk}")
            st.caption(f"顯著水準 α = {config.ALPHA}，預測區間信心水準 = {config.PI_CONFIDENCE:.0%}")
            settings = {
                "test_size": float(test_size),
                "random_state": int(random_state),
                "cv_folds": int(cv_folds),
                "k": int(k_value),
                "direction": direction,
                "final_method": final_method,
                "alpha": config.ALPHA,
            }

    run_clicked = st.button(
        "執行分析",
        type="primary",
        disabled=cleaned is None or not features or settings is None,
        key=f"run_{pk}",
    )

# 本次設定的簽章：用於判斷上次結果是否仍符合目前設定
data_hash = None
signature = None
if cleaned is not None and settings is not None and features:
    data_hash = int(pd.util.hash_pandas_object(cleaned, index=False).sum())
    signature = (
        problem,
        source,
        url.strip(),
        getattr(upload, "name", ""),
        target,
        tuple(features),
        data_hash,
        tuple(sorted(settings.items())),
    )

auto_run = signature is not None and not st.session_state.get("auto_run_done", False)
if auto_run:
    st.session_state["auto_run_done"] = True

if (run_clicked or auto_run) and signature is not None:
    try:
        with st.spinner("正在執行分析；若使用 SFS 且特徵較多，可能需要一些時間…"):
            result = cached_analysis(cleaned, target, tuple(features), settings)
        st.session_state["results"] = {
            "problem": problem,
            "signature": signature,
            "result": result,
            "source_desc": source_desc,
            "cleaned": cleaned,
            "clean_report": clean_report,
            "target": target,
        }
        st.success("分析完成。")
    except ValueError as exc:
        st.error(f"無法完成分析：{exc}")
    except Exception as exc:  # noqa: BLE001 - 避免整個網站崩潰
        st.error(f"分析過程發生錯誤：{type(exc).__name__}: {exc}")

stored = st.session_state.get("results")
result = None
if stored and stored["problem"] == problem:
    result = stored["result"]
    if stored["signature"] != signature:
        st.warning("目前的設定或資料已與上次執行的結果不同。以下為上次執行的結果，請重新執行分析。")
elif stored and stored["problem"] != problem:
    st.info("已切換 Problem，請於側邊欄設定後重新執行分析。")

if result is not None:
    source_desc = stored["source_desc"]
    cleaned_for_results = stored["cleaned"]
    clean_report_for_results = stored["clean_report"]
    target_for_results = stored["target"]
    charts = result["charts"]
else:
    cleaned_for_results = cleaned
    clean_report_for_results = clean_report
    target_for_results = target
    charts = {}

charts_da = (
    data_understanding_charts(cleaned, target)
    if cleaned is not None and target is not None
    else {}
)

tabs = st.tabs(TAB_NAMES)
with tabs[0]:
    render_intro(problem)
with tabs[1]:
    render_business(problem, target, features)
with tabs[2]:
    render_data_understanding(raw, cleaned, target, source_desc, charts_da)
with tabs[3]:
    render_preparation(
        cleaned,
        clean_report,
        target,
        features,
        settings["test_size"] if settings else config.DEFAULT_TEST_SIZE,
        settings["random_state"] if settings else config.DEFAULT_RANDOM_STATE,
    )
with tabs[4]:
    render_selection(result)
with tabs[5]:
    render_evaluation(result)
with tabs[6]:
    render_assumptions(result)
with tabs[7]:
    render_deployment(result, cleaned_for_results, target_for_results)
with tabs[8]:
    render_ai(problem)
with tabs[9]:
    render_downloads(result, problem, source_desc, charts)
