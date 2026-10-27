"""下載檔與 Markdown 報告產生模組。

所有數值皆取自實際執行的分析結果，不包含任何預先填入的假數字。
"""

from __future__ import annotations

import math

import pandas as pd

from . import ai_content
from .feature_selection import display_name


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """輸出 CSV（UTF-8 with BOM，確保 Excel 開啟繁體中文不亂碼）。"""
    return df.to_csv(index=False).encode("utf-8-sig")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if math.isinf(value):
            return "無限大"
        return f"{value:.{digits}f}"
    return str(value)


def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """不依賴 tabulate 的 Markdown 表格輸出。"""
    if df is None or df.empty:
        return "（無資料）"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = [_fmt(v, digits).replace("|", "\\|").replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_export_tables(result: dict) -> dict[str, pd.DataFrame]:
    """整理六份可下載的 CSV 表格。"""
    metrics = result["final_metrics"]
    metrics_df = pd.DataFrame(
        [
            {"指標": "MAE", "數值": metrics["MAE"]},
            {"指標": "MSE", "數值": metrics["MSE"]},
            {"指標": "RMSE", "數值": metrics["RMSE"]},
            {"指標": "R²", "數值": metrics["R²"]},
            {"指標": "Adjusted R²", "數值": metrics["Adjusted R²"]},
            {"指標": "測試集樣本數 n", "數值": metrics["n_test"]},
            {"指標": "有效預測變數數 p", "數值": metrics["p_predictors"]},
            {"指標": "最終模型", "數值": result["final_method"]},
        ]
    )

    selected_df = pd.DataFrame(
        {
            "轉換後特徵": result["final_selected"],
            "顯示名稱": [display_name(n) for n in result["final_selected"]],
        }
    )

    assumptions_df = result["assumptions"]["hypothesis_table"].copy()

    pred_df = result["predictions"].copy()

    comparison_df = result["comparison_table"].copy()

    selected_by_method = pd.DataFrame(
        [
            {
                "特徵選擇方法": m["method"],
                "特徵數量": m["n_selected"] if m["n_selected"] is not None else "失敗",
                "選中的轉換後特徵": "；".join(m["selected"]) if m["selected"] else (m["error"] or ""),
            }
            for m in result["method_results"]
        ]
    )

    return {
        "方法比較.csv": comparison_df,
        "最終模型指標.csv": metrics_df,
        "選中特徵清單.csv": selected_df,
        "假設檢驗結果.csv": assumptions_df,
        "預測值與實際值.csv": pred_df,
        "各方法選中特徵.csv": selected_by_method,
    }


def build_markdown_report(
    result: dict,
    problem_label: str,
    source_description: str,
    include_ai_sections: bool = True,
) -> str:
    """依實際分析結果產生完整 Markdown 報告。"""
    s = result
    m = s["final_metrics"]
    a = s["assumptions"]
    h = a["hypothesis_table"]
    lines: list[str] = []
    add = lines.append

    add("# 多元線性迴歸分析報告")
    add("")
    add(f"- 分析問題：{problem_label}")
    add(f"- 資料來源：{source_description}")
    add(f"- 目標變數：{s['target']}")
    add(f"- 原始資料筆數：{s['n_total']}")
    add(f"- 訓練集筆數：{s['n_train']}；測試集筆數：{s['n_test']}")
    add(f"- 使用者選擇的輸入欄位數：{len(s['features_input'])}")
    add(f"- 轉換後（含 One-Hot）特徵數：{s['n_transformed_features']}")
    add(f"- 最終模型特徵選擇方法：{s['final_method']}")
    add("")

    add("## 1. 研究目的")
    add("")
    add(f"本報告以多元線性迴歸分析「{s['target']}」，比較五種特徵選擇方法，並檢驗迴歸假設與預測區間。")
    if s["effective_feature_warning"]:
        add("")
        add(
            f"注意：本資料集轉換後只有 {s['n_transformed_features']} 個有效特徵，"
            "少於本作業要求的 10 個特徵，結果不一定符合作業的特徵數量要求。"
        )
    add("")

    add("## 2. 資料來源與資料理解")
    add("")
    add(f"- 資料來源：{source_description}")
    add(f"- 使用的數值特徵：{', '.join(s['numeric_cols']) or '無'}")
    add(f"- 使用的類別特徵：{', '.join(s['categorical_cols']) or '無'}")
    if s["skipped_cols"]:
        add(f"- 因型態不適用而排除的欄位：{', '.join(s['skipped_cols'])}")
    add("")

    add("## 3. 資料準備")
    add("")
    add("- 移除重複資料、排除無法轉換的欄位、丟棄目標變數缺失的列。")
    add("- 數值特徵：中位數填補缺失值後標準化（StandardScaler）。")
    add("- 類別特徵：固定填補為缺失類別後進行 One-Hot Encoding（drop='first'）。")
    add("- 所有前處理與特徵選擇只在訓練集上擬合，測試集僅用於最終評估。")
    add("")

    add("## 4. 五種特徵選擇方法")
    add("")
    for entry in s["method_results"]:
        add(f"### {entry['method']}")
        add("")
        add(entry["description"])
        add("")
        if entry["error"]:
            add(f"執行失敗：{entry['error']}")
        else:
            add(f"- 選擇數量：{entry['n_selected']}")
            add(f"- 選中的特徵：{', '.join(display_name(f) for f in entry['selected'])}")
            for note in entry["notes"]:
                add(f"- 說明：{note}")
        add("")

    add("## 5. 模型評估結果")
    add("")
    add(df_to_markdown(s["comparison_table"]))
    add("")
    add(
        "注意：Adjusted R² 公式為 Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)，"
        "其中 n 為測試集樣本數，p 為最終模型實際使用的轉換後特徵數（不含截距）。"
    )
    add("R² 描述模型在測試集上解釋變異的比例；Adjusted R² 會依有效預測變數數量進行懲罰。")
    add("")
    add(f"最終模型（{s['final_method']}）的指標：")
    add("")
    add(f"- MAE：{_fmt(m['MAE'])}")
    add(f"- MSE：{_fmt(m['MSE'])}")
    add(f"- RMSE：{_fmt(m['RMSE'])}")
    add(f"- R²：{_fmt(m['R²'])}")
    add(f"- Adjusted R²：{_fmt(m['Adjusted R²'])}（{m['Adjusted R² 說明']}）")
    add("")

    add("## 6. 迴歸假設檢驗")
    add("")
    add(df_to_markdown(h))
    add("")
    add(
        "統計檢定的 p-value 不是「假設成立的機率」。未拒絕虛無假設只代表未發現足夠證據拒絕該假設，"
        "不代表假設已被證明成立。Shapiro-Wilk 對大樣本的輕微偏離很敏感，請同時參考 Q-Q Plot。"
    )
    add("")

    add("## 7. 預測圖與 95% 預測區間")
    add("")
    add(
        "預測區間使用 statsmodels OLS 的 get_prediction().summary_frame()，"
        "描述個別新觀測值的不確定性，通常比平均反應的信賴區間寬。"
    )
    add("")
    add(f"OLS 與 Sklearn 預測值的最大絕對差異：{_fmt(s['sklearn_vs_ols_max_diff'], 8)}")
    add("")

    add("## 8. CRISP-DM 六階段")
    add("")
    for stage in [
        "1. 商業理解：研究汽車規格與售價之間的關係，或使用者自訂的預測目標。",
        "2. 資料理解：檢查資料筆數、欄位型態、缺失值、重複資料與描述性統計。",
        "3. 資料準備：清理資料、前處理、特徵選擇與訓練／測試切分。",
        "4. 建模：以 LinearRegression 建立多元線性迴歸模型，並比較五種特徵選擇方法。",
        "5. 評估：比較 MAE、MSE、RMSE、R²、Adjusted R²，並進行假設檢驗與預測區間分析。",
        "6. 部署：以 Streamlit 展示，說明預測結果不代表實際成交價格，且僅適用於與訓練資料相近的情境。",
    ]:
        add(f"- {stage}")
    add("")

    if include_ai_sections:
        add("## 9. GPT 輔助內容")
        add("")
        for item in ai_content.GPT_ASSIST_ITEMS:
            add(f"- {item}")
        add("")
        add(ai_content.GPT_DISCLAIMER)
        add("")
        add(f"## 10. {ai_content.NOTEBOOKLM_DRAFT_LABEL}")
        add("")
        add(ai_content.NOTEBOOKLM_DRAFT_TEXT)
        add("")
        add("參考來源：")
        for name, kind, url in ai_content.RESEARCH_SOURCES:
            add(f"- {name}（{kind}）：{url}")
        add("")

    add("## 11. 研究限制與結論")
    add("")
    add("- 線性模型只能捕捉線性關係，無法自動處理複雜的非線性效果。")
    add("- 相關性與係數不代表因果關係。")
    add("- 模型只適用於與訓練資料特性相近的資料，預測結果不代表實際成交價格。")
    add("- Durbin-Watson 主要檢查特定形式的殘差自相關；橫斷面資料不能僅憑此檢定宣稱誤差獨立。")
    add("- 測試集樣本數有限時，各指標與方法排名可能受抽樣影響。")
    add("")
    return "\n".join(lines)
