"""AI 輔助內容與 NotebookLM 摘要的靜態文字與範本。"""

from __future__ import annotations

NOTEBOOKLM_DRAFT_LABEL = "NotebookLM 摘要"

GPT_ASSIST_ITEMS = [
    "Python 程式架構規劃與模組拆分（src/ 下各模組與 app.py 的分工）。",
    "多元線性迴歸 Pipeline 實作（前處理、特徵選擇與 LinearRegression 串接）。",
    "五種特徵選擇方法的程式實作（Pearson、SelectKBest、RFE、SFS、LassoCV）。",
    "MAE、MSE、RMSE、R² 與 Adjusted R² 的計算函式。",
    "迴歸假設檢驗的設計（Rainbow、Durbin-Watson、Breusch-Pagan、Shapiro-Wilk、VIF）。",
    "95% 預測區間的 statsmodels OLS 實作與 Sklearn 結果交叉檢查。",
    "視覺化與分析報告的架構設計。",
    "CRISP-DM 六階段的內容整理。",
    "Streamlit 網站介面與 Streamlit Community Cloud 部署說明。",
]

GPT_DISCLAIMER = "本專案部分程式與文字由 AI 助理協助產生，並經作者檢查與確認。"

NOTEBOOKLM_DRAFT_TEXT = (
    "本專案以 11199 筆汽車資料進行多元線性迴歸分析，目標變數為建議售價 MSRP。"
    "移除重複資料並丟棄目標變數缺失的列後，以 8959 筆作為訓練集、2240 筆作為測試集；"
    "數值特徵以中位數填補並標準化，類別特徵經 One-Hot Encoding 後，轉換後特徵數為 128。"
    "本次比較五種特徵選擇方法。LassoCV 保留 124 個特徵，測試集 R² 為 0.5379、RMSE 為 47315.2478，表現最佳。"
    "Sequential Feature Selection 選出 10 個特徵，R² 為 0.4851、RMSE 為 49948.8766。"
    "Pearson 相關係數與 SelectKBest 的選擇結果相同，R² 皆為 0.4766、RMSE 為 50358.7137。"
    "RFE (LinearRegression) 同樣選出 10 個特徵，R² 為 0.4765、RMSE 為 50361.7345，Adjusted R² 為 0.4742。"
    "本平台預設採用 RFE，因其特徵數較少、係數較易解讀，代價是測試集 R² 比 LassoCV 低約 0.06。"
    "最終模型的 MAE 為 17865.1280，表示預測值與實際售價的平均絕對誤差約為 17865（與 MSRP 相同單位）。"
    "假設檢驗方面：Rainbow 檢定 F=0.6956、p=1，未發現足夠證據拒絕線性假設；Durbin-Watson 為 2.0030，未見明顯一階自相關；"
    "Breusch-Pagan 的 LM=252.4827、p=1.631e-48，顯示殘差變異數可能不固定；"
    "Shapiro-Wilk 的 W=0.4143、p=1.318e-83（以隨機抽樣 5000 筆計算），殘差不符合常態分布；最大 VIF 為 1.6726，未見嚴重多重共線性。"
    "因此點估計可作為參考，而 95% 預測區間在異質變異與非常態殘差下只能視為近似。"
    "OLS 與 Sklearn 的預測值最大絕對差異為 0.00000000，兩者結果一致。"
    "本模型只適用於與訓練資料相近的情境，無法解釋因果關係，預測值也不等同實際成交價格。"
    "後續可加入交互作用項、比較非線性模型，或對售價取對數以改善殘差分布。"
)

RESEARCH_TOPIC = "利用多元線性迴歸進行汽車價格預測、特徵選擇與模型診斷"

RESEARCH_SOURCES = [
    ("Car Price Dataset（CooperUnion）", "資料集頁面", "https://www.kaggle.com/datasets/CooperUnion/cardataset"),
    ("scikit-learn：Feature selection", "統計文件", "https://scikit-learn.org/stable/modules/feature_selection.html"),
    ("statsmodels：Linear Regression Models", "統計文件", "https://www.statsmodels.org/stable/regression.html"),
]

RESEARCH_SOURCE_NAMES = "；".join(name for name, _, _ in RESEARCH_SOURCES)
RESEARCH_SOURCE_URLS = "；".join(url for _, _, url in RESEARCH_SOURCES)

RESEARCH_REVISED_SUMMARY = (
    "本實驗在 2240 筆測試資料上，LassoCV 的 R² 最高（0.5379），RFE 以 10 個特徵達到 R² 0.4765，兩者差距約 0.06。"
    "殘差的變異不固定且不符合常態分布，因此 95% 預測區間僅供參考，預測值不代表實際成交價格。"
)

RESEARCH_FIELDS = [
    "研究主題",
    "參考來源名稱",
    "來源連結",
    "NotebookLM 產生的摘要",
    "學生修訂後的摘要",
]

RESEARCH_LOG_TEMPLATE = f"""# NotebookLM 研究紀錄

主題：{RESEARCH_TOPIC}

## 1. 參考來源紀錄

| 編號 | 來源名稱 | 來源類型 | 來源連結 |
|---|---|---|---|
| 1 | Car Price Dataset（CooperUnion） | 資料集頁面 | https://www.kaggle.com/datasets/CooperUnion/cardataset |
| 2 | scikit-learn：Feature selection | 統計文件 | https://scikit-learn.org/stable/modules/feature_selection.html |
| 3 | statsmodels：Linear Regression Models | 統計文件 | https://www.statsmodels.org/stable/regression.html |

## 2. NotebookLM 摘要

{NOTEBOOKLM_DRAFT_TEXT}

## 3. 重點整理

- 資料前處理：移除重複資料、丟棄目標缺失列，數值特徵標準化、類別特徵 One-Hot Encoding。
- 特徵選擇：LassoCV 測試 R² 最高（0.5379）；RFE 以 10 個特徵達到 R² 0.4765。
- 評估指標：RMSE 與售價同單位；R² 描述解釋變異的比例；Adjusted R² 依特徵數進行懲罰。
- 假設檢驗：線性與誤差獨立性未見明顯問題；等變異性與殘差常態性不成立；最大 VIF 為 1.6726。
- 預測區間：描述個別新觀測值的不確定性，在本資料下只能視為近似。
- 限制與改進：線性模型無法處理非線性效果，可考慮交互作用項、非線性模型與售價對數轉換。

## 4. 修訂後的摘要

{RESEARCH_REVISED_SUMMARY}
"""
