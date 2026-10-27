"""迴歸五項假設檢驗模組。

所有檢定皆以最終模型在「訓練資料」上的 OLS 殘差進行診斷。
統計檢定的 p-value 不代表假設成立的機率；未拒絕虛無假設只代表「未發現足夠證據拒絕」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from . import config


def linearity_test(ols_results) -> dict:
    """線性關係：Rainbow 檢定（虛無假設為線性關係成立）。"""
    try:
        f_stat, p_value = linear_rainbow(ols_results)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Rainbow 檢定無法計算：{exc}"}
    return {"統計量": float(f_stat), "p-value": float(p_value), "統計量名稱": "F"}


def independence_test(residuals: np.ndarray) -> dict:
    """誤差獨立性：Durbin-Watson 統計量（約 2 代表較少的一階自相關證據）。"""
    dw = float(durbin_watson(np.asarray(residuals, dtype=float)))
    return {"統計量": dw, "統計量名稱": "Durbin-Watson"}


def homoscedasticity_test(residuals: np.ndarray, exog: np.ndarray) -> dict:
    """誤差等變異性：Breusch-Pagan 檢定（exog 需含常數項）。"""
    try:
        lm, lm_p, f_stat, f_p = het_breuschpagan(np.asarray(residuals, dtype=float), np.asarray(exog, dtype=float))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Breusch-Pagan 檢定無法計算：{exc}"}
    return {
        "LM 統計量": float(lm),
        "LM p-value": float(lm_p),
        "F 統計量": float(f_stat),
        "F p-value": float(f_p),
        "p-value": float(lm_p),
        "統計量名稱": "LM",
    }


def normality_test(residuals: np.ndarray, random_state: int = config.DEFAULT_RANDOM_STATE) -> dict:
    """殘差常態性：Shapiro-Wilk 檢定。樣本超過 5000 時改用隨機抽樣 5000 筆並說明。"""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {"error": "殘差樣本數不足 3 筆，無法執行 Shapiro-Wilk 檢定。"}
    note = ""
    if len(values) > config.SHAPIRO_MAX_N:
        rng = np.random.RandomState(random_state)
        values = rng.choice(values, size=config.SHAPIRO_MAX_N, replace=False)
        note = (
            f"殘差樣本數超過 {config.SHAPIRO_MAX_N}，Shapiro-Wilk 改以隨機抽樣 {config.SHAPIRO_MAX_N} 筆計算；"
            "樣本很大時此檢定對輕微偏離非常敏感，請同時觀察 Q-Q Plot。"
        )
    w_stat, p_value = stats.shapiro(values)
    return {"統計量": float(w_stat), "p-value": float(p_value), "統計量名稱": "W", "note": note}


def multicollinearity_table(design: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """計算各特徵的 VIF（design 不含常數項，函式內部加入常數項）。

    Returns:
        (VIF 表, 說明訊息列表)
    """
    notes: list[str] = []
    if design.shape[1] == 0:
        return pd.DataFrame(columns=["特徵", "VIF", "判斷"]), ["沒有可計算 VIF 的特徵。"]

    exog = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    rank = np.linalg.matrix_rank(exog)
    if rank < exog.shape[1]:
        notes.append(
            f"設計矩陣秩為 {rank}，小於欄位數 {exog.shape[1]}，代表存在完全共線性（例如冗餘的虛擬變數）。"
            "此時部分 VIF 會趨近無限大，應檢視類別變數與特徵組合。"
        )

    rows = []
    for idx, name in enumerate(design.columns, start=1):
        column = exog[:, idx]
        if np.nanstd(column) == 0:
            rows.append({"特徵": name, "VIF": np.nan, "判斷": "常數欄位，VIF 無法計算"})
            notes.append(f"特徵「{name}」為常數欄位，VIF 無法計算。")
            continue
        with np.errstate(all="ignore"):
            value = float(variance_inflation_factor(exog, idx))
        if not np.isfinite(value):
            judgement = "無限大（完全共線）"
        elif value > config.VIF_THRESHOLD:
            judgement = f"> {config.VIF_THRESHOLD:g}，需進一步注意"
        else:
            judgement = f"≤ {config.VIF_THRESHOLD:g}"
        rows.append({"特徵": name, "VIF": value, "判斷": judgement})
    table = pd.DataFrame(rows)
    return table, notes


def run_assumption_tests(
    ols_results,
    design: pd.DataFrame,
    random_state: int = config.DEFAULT_RANDOM_STATE,
    alpha: float = config.ALPHA,
) -> dict:
    """執行五項假設檢驗，回傳各項結果與總表。"""
    residuals = np.asarray(ols_results.resid, dtype=float)
    exog = np.asarray(ols_results.model.exog, dtype=float)

    linearity = linearity_test(ols_results)
    independence = independence_test(residuals)
    homoscedasticity = homoscedasticity_test(residuals, exog)
    normality = normality_test(residuals, random_state=random_state)
    vif_df, vif_notes = multicollinearity_table(design)

    max_vif = float(np.nanmax(vif_df["VIF"].to_numpy(dtype=float))) if vif_df["VIF"].notna().any() else float("nan")

    summary_rows = []

    # 1. 線性關係
    if "error" in linearity:
        judgement, result = "無法判斷", linearity["error"]
    else:
        p = linearity["p-value"]
        result = f"F={linearity['統計量']:.4f}，p={p:.4g}"
        if p < alpha:
            judgement = "有證據顯示可能存在非線性"
        else:
            judgement = "未發現足夠證據拒絕線性假設"
    summary_rows.append(
        {
            "假設": "1. 線性關係",
            "檢查方法": "Rainbow 檢定 + 殘差對擬合值圖",
            "結果": result,
            "判斷": judgement,
            "說明": "Rainbow 檢定的虛無假設為線性關係成立；仍需檢視殘差圖是否有彎曲趨勢。",
        }
    )

    # 2. 誤差獨立性
    dw = independence["統計量"]
    if 1.5 <= dw <= 2.5:
        judgement = "未發現明顯一階自相關證據"
    else:
        judgement = "DW 偏離 2，需檢查資料順序與依賴結構"
    summary_rows.append(
        {
            "假設": "2. 誤差獨立性",
            "檢查方法": "Durbin-Watson",
            "結果": f"DW={dw:.4f}",
            "判斷": judgement,
            "說明": "DW 接近 2 代表較少的一階自相關證據（1.5～2.5 為常用經驗範圍）。"
            "本檢定主要針對順序或時間相關的殘差；橫斷面資料不可僅憑此檢定就宣稱誤差完全獨立。",
        }
    )

    # 3. 誤差等變異性
    if "error" in homoscedasticity:
        judgement, result = "無法判斷", homoscedasticity["error"]
    else:
        p = homoscedasticity["LM p-value"]
        result = f"LM={homoscedasticity['LM 統計量']:.4f}，p={p:.4g}"
        if p < alpha:
            judgement = "有證據顯示殘差變異數可能不固定"
        else:
            judgement = "未發現足夠證據拒絕等變異性"
    summary_rows.append(
        {
            "假設": "3. 誤差等變異性",
            "檢查方法": "Breusch-Pagan + 尺度位置圖",
            "結果": result,
            "判斷": judgement,
            "說明": "單一檢定不能完全排除或證明異質變異，需搭配殘差圖一起判讀。",
        }
    )

    # 4. 殘差常態性
    if "error" in normality:
        judgement, result = "無法判斷", normality["error"]
    else:
        p = normality["p-value"]
        result = f"W={normality['統計量']:.4f}，p={p:.4g}"
        if p < alpha:
            judgement = "有足夠證據拒絕殘差常態性"
        else:
            judgement = "未發現足夠證據拒絕殘差常態性"
    summary_rows.append(
        {
            "假設": "4. 殘差常態性",
            "檢查方法": "Shapiro-Wilk + 直方圖 + Q-Q Plot",
            "結果": result,
            "判斷": judgement,
            "說明": normality.get("note") or "請同時參考 Q-Q Plot；常態性主要影響推論區間的精確度。",
        }
    )

    # 5. 多重共線性
    if np.isnan(max_vif):
        judgement, result = "無法判斷", "沒有可計算的 VIF"
    else:
        result = f"最大 VIF={max_vif:.4f}" if np.isfinite(max_vif) else "最大 VIF 為無限大"
        if (not np.isfinite(max_vif)) or max_vif > config.VIF_THRESHOLD:
            judgement = "需進一步注意多重共線性"
        else:
            judgement = "未見 VIF 超過 5 的特徵"
    summary_rows.append(
        {
            "假設": "5. 無嚴重多重共線性",
            "檢查方法": "Variance Inflation Factor (VIF)",
            "結果": result,
            "判斷": judgement,
            "說明": "VIF > 5 是常見的經驗提示門檻，並非絕對標準；類別變數的虛擬變數若完全共線會造成無限大 VIF。",
        }
    )

    hypothesis_df = pd.DataFrame(summary_rows)
    return {
        "linearity": linearity,
        "independence": independence,
        "homoscedasticity": homoscedasticity,
        "normality": normality,
        "vif_table": vif_df,
        "vif_notes": vif_notes,
        "residuals": residuals,
        "fitted": np.asarray(ols_results.fittedvalues, dtype=float),
        "exog": exog,
        "hypothesis_table": hypothesis_df,
        "alpha": alpha,
    }
