# 多元線性迴歸分析平台

以 Streamlit 建置的多元線性迴歸分析網站，流程遵循 CRISP-DM 六階段，包含兩個 Problem：

- **Problem 1：汽車價格預測**：Kaggle 公開資料集 Car Features and MSRP（CooperUnion/cardataset），預測目標為 MSRP。
- **Problem 2：自訂資料集迴歸分析**：可輸入公開 CSV 網址、上傳 CSV，或使用 Startup 50 範例資料集（預設目標為 Profit）。

## 功能說明

- 資料理解：基本資訊、缺失值、重複值、描述性統計、相關係數矩陣與圖表。
- 資料準備：重複值移除、型態轉換、缺失值填補、標準化、One-Hot Encoding，全部包在 sklearn Pipeline 中，只在訓練集上擬合。
- 特徵選擇：Pearson 相關係數、SelectKBest（f_regression）、RFE（LinearRegression）、Sequential Feature Selection（前向／後向）、LassoCV（L1 正則化）。
- 模型評估：MAE、MSE、RMSE、R²、Adjusted R²（皆以測試集計算），並比較五種方法。
- 預測區間：以 statsmodels OLS 計算個別新觀測值的 95% 預測區間。
- 迴歸假設檢驗：線性關係（Rainbow 檢定與殘差圖）、誤差獨立性（Durbin-Watson）、等變異性（Breusch-Pagan）、常態性（Shapiro-Wilk 與 Q-Q Plot）、多重共線性（VIF）。
- 部署與預測：單筆資料預測值與 95% 預測區間。
- AI 輔助與研究摘要：GPT 輔助內容說明、NotebookLM 研究紀錄範本（摘要草稿需自行研究驗證）。
- 報告與下載：六份 CSV、Markdown 完整報告、PNG 圖表。

## 資料來源

### Problem 1：汽車資料

- 來源：<https://www.kaggle.com/datasets/CooperUnion/cardataset>
- Kaggle 需要登入才能下載，本專案不會自動下載或產生假資料。請自行下載 CSV，並選擇以下任一方式：
  1. 將檔案另存為 `data/data.csv`（詳見 [data/README.md](data/README.md)）。
  2. 在網站側邊欄選擇「上傳 CSV 檔案」。
  3. 在網站側邊欄選擇「公開 CSV 網址」，輸入可公開讀取的 CSV 連結。

### Problem 2：自訂資料

- Startup 50：<https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv>（需要網路連線）。
- 也可使用任意公開 CSV 網址或上傳本機 CSV。

## 安裝步驟

需要 Python 3.10 以上版本。

```bash
python -m pip install -r requirements.txt
```

若系統同時安裝多個 Python，建議使用 `python -m pip`，確保套件安裝到執行程式的同一個直譯器。

## 本機執行方式

在專案根目錄執行：

```bash
python -m streamlit run app.py
```

瀏覽器會開啟 `http://localhost:8501`。

在 VS Code 中也可開啟終端機，於專案資料夾執行上述指令。

### 預先建置的預設分析結果

網站首次開啟時會直接讀取 `precomputed/default_analysis.json`（預設汽車資料與預設設定的分析結果與圖表），不必重新分析。此檔已納入版本控制。

若修改 `data/data.csv`、目標變數、特徵或預設設定，請重新產生快照：

```bash
python scripts/build_default_snapshot.py
```

快照與目前的資料指紋或設定不符時，網站會自動改為即時分析。

## 如何使用自訂 CSV URL

1. 在頁面上方選擇 Problem。
2. 在側邊欄資料來源選擇「公開 CSV 網址」（Problem 1）或「公開 CSV 網址」（Problem 2）。
3. 輸入以 `http://` 或 `https://` 開頭、可直接下載 CSV 的連結（HTML 頁面無法使用）。
4. 系統會嘗試以 UTF-8、UTF-8-SIG、Big5、CP950 解碼；檔案上限 100 MB，下載逾時為 30 秒。

## 如何執行五種特徵選擇

1. 在側邊欄選擇目標變數與輸入特徵（ID 類欄位預設排除，可自行調整）。
2. 設定測試集比例、Random State、交叉驗證折數與 K 值。
3. 選擇 SFS 方向（前向或後向）與最終模型採用的特徵選擇方法。
4. 按下「執行分析」。系統會依序執行五種方法；單一方法失敗時會顯示錯誤，其餘方法仍會完成。

所有特徵選擇只使用訓練集；測試集只用於最終評估。

## 如何查看模型評估結果

- 「評估與預測區間」分頁：最終模型的五項指標、實際值與預測值圖、95% 預測區間、OLS 係數表與測試集預測結果。
- 「建模與特徵選擇」分頁：五種方法的選中特徵、分數與測試集指標比較表（可點擊欄位標題排序）。

## 如何查看迴歸假設檢驗

「迴歸假設檢驗」分頁提供假設檢驗總表、殘差圖、Q-Q Plot、VIF 表與 VIF 長條圖。未拒絕虛無假設只代表「未發現足夠證據拒絕」，不代表假設已被證明成立。

## GitHub 上傳方式

```bash
git init
git add .
git commit -m "Add multiple linear regression Streamlit app"
git branch -M main
git remote add origin https://github.com/<你的帳號>/<專案名稱>.git
git push -u origin main
```

`.gitignore` 僅排除虛擬環境、Python 快取與本機分析快取（`cache/`）。`data/` 內的 CSV 與 `reports/` 的說明檔會一併提交至 GitHub。

## Streamlit Community Cloud 部署方式

1. 將專案推送到 GitHub 公開儲存庫。
2. 登入 <https://share.streamlit.io>，選擇 New app。
3. Repository 選擇本專案，Branch 選擇 `main`，Main file path 填入 `app.py`。
4. 部署時會自動讀取 `requirements.txt` 安裝 Python 套件，並讀取 `packages.txt` 安裝系統字型（fonts-noto-cjk，用於中文圖表）。
5. 若要使用 Problem 1 的本機資料，請確認 `data/data.csv` 已一併提交；未提交時請改用「上傳 CSV 檔案」或「公開 CSV 網址」。

本專案不需要任何 API 金鑰或 secrets。

## 專案結構

```text
HW2/
├── app.py                         Streamlit 入口
├── requirements.txt               Python 套件
├── packages.txt                   Streamlit Cloud 系統套件（中文字型）
├── .gitignore
├── README.md
├── prompt.md                      作業規格
├── data/
│   └── README.md                  Kaggle 資料放置說明
├── reports/
│   ├── README.md
│   └── notebooklm_research_log_template.md
└── src/
    ├── config.py                  共用設定
    ├── data_loader.py             CSV 讀取（本機、URL、上傳、Startup 50）
    ├── data_preprocessing.py      資料清理與前處理
    ├── feature_selection.py       五種特徵選擇方法
    ├── regression_model.py        Pipeline、OLS 與預測區間
    ├── evaluation.py              指標計算
    ├── assumption_tests.py        迴歸假設檢驗
    ├── analysis_runner.py         完整分析流程
    ├── visualization.py           圖表（PNG）
    ├── ai_content.py              GPT 輔助與 NotebookLM 草稿內容
    └── report_generator.py        CSV 與 Markdown 報告
```

## 已知限制

- 本專案不包含 Kaggle 汽車資料，請依上方說明自行提供。
- Sequential Feature Selection 會對每個候選特徵進行交叉驗證，特徵數量多時（例如汽車資料 One-Hot 後約 190 個特徵）可能需要數分鐘；後向選擇在轉換後特徵超過 30 個時會自動改用前向選擇，並於結果中說明。
- 類別變數的類別數超過 30 時，只保留前 30 個類別，其餘歸為稀少類別。
- Startup 50 原始只有 4 個輸入欄位，轉換後約 5 個特徵，少於作業要求的 10 個有效特徵，平台會明確提示。
- Streamlit Community Cloud 上的中文字型顯示效果尚未在雲端環境實際驗證。
- 「NotebookLM 摘要草稿」僅為待研究驗證的草稿，尚未對實際參考來源進行研究，不可直接作為文獻結論。
- 線性模型只能捕捉線性關係，相關性不代表因果關係。
