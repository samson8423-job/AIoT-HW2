# 多元線性迴歸分析報告

- 分析問題：Problem 1：汽車價格預測（Car Price Prediction）
- 資料來源：本機檔案 data/data.csv
- 目標變數：MSRP
- 原始資料筆數：11199
- 訓練集筆數：8959；測試集筆數：2240
- 使用者選擇的輸入欄位數：15
- 轉換後（含 One-Hot）特徵數：128
- 最終模型特徵選擇方法：RFE (LinearRegression)

## 1. 研究目的

本報告以多元線性迴歸分析「MSRP」，比較五種特徵選擇方法，並檢驗迴歸假設與預測區間。

## 2. 資料來源與資料理解

- 資料來源：本機檔案 data/data.csv
- 使用的數值特徵：Year, Engine HP, Engine Cylinders, Number of Doors, highway MPG, city mpg, Popularity
- 使用的類別特徵：Make, Model, Engine Fuel Type, Transmission Type, Driven_Wheels, Market Category, Vehicle Size, Vehicle Style

## 3. 資料準備

- 移除重複資料、排除無法轉換的欄位、丟棄目標變數缺失的列。
- 數值特徵：中位數填補缺失值後標準化（StandardScaler）。
- 類別特徵：固定填補為缺失類別後進行 One-Hot Encoding（drop='first'）。
- 所有前處理與特徵選擇只在訓練集上擬合，測試集僅用於最終評估。

## 4. 五種特徵選擇方法

### Pearson 相關係數

依訓練資料中每個轉換後特徵與目標變數的 Pearson 相關係數絕對值排序，取前 K 個。

- 選擇數量：10
- 選中的特徵：Engine HP, Engine Cylinders, Make_infrequent_sklearn, Engine Fuel Type_premium unleaded (required), Engine Fuel Type_regular unleaded, Driven_Wheels_front wheel drive, Market Category_Exotic,High-Performance, Market Category_Exotic,Luxury,High-Performance, Market Category___缺失__, Market Category_infrequent_sklearn

### SelectKBest (f_regression)

以 f_regression 計算每個特徵與目標的線性 F 分數，取分數最高的前 K 個。

- 選擇數量：10
- 選中的特徵：Engine HP, Engine Cylinders, Make_infrequent_sklearn, Engine Fuel Type_premium unleaded (required), Engine Fuel Type_regular unleaded, Driven_Wheels_front wheel drive, Market Category_Exotic,High-Performance, Market Category_Exotic,Luxury,High-Performance, Market Category___缺失__, Market Category_infrequent_sklearn

### RFE (LinearRegression)

以 LinearRegression 的係數大小反覆移除最不重要的特徵，直到剩下 K 個。

- 選擇數量：10
- 選中的特徵：Engine HP, Make_infrequent_sklearn, Engine Fuel Type_flex-fuel (premium unleaded recommended/E85), Engine Fuel Type_natural gas, Market Category_Exotic,Factory Tuner,Luxury,High-Performance, Market Category_Exotic,High-Performance, Market Category_Exotic,Luxury,High-Performance, Market Category_Factory Tuner,High-Performance, Market Category_High-Performance, Market Category_infrequent_sklearn

### Sequential Feature Selection

以 LinearRegression 搭配交叉驗證（RMSE），逐步加入（前向）或移除（後向）特徵，直到剩下 K 個。

- 選擇數量：10
- 選中的特徵：Engine HP, Make_infrequent_sklearn, Engine Fuel Type_premium unleaded (required), Driven_Wheels_front wheel drive, Market Category_Exotic,Factory Tuner,Luxury,High-Performance, Market Category_Exotic,High-Performance, Market Category_Exotic,Luxury,High-Performance, Market Category_Factory Tuner,High-Performance, Market Category_High-Performance, Market Category_infrequent_sklearn

### LassoCV (L1 正則化)

以 LassoCV（交叉驗證選 alpha）在標準化後的特徵上擬合，保留係數非零的特徵，特徵數由資料決定。

- 選擇數量：124
- 選中的特徵：Year, Engine HP, Engine Cylinders, Number of Doors, city mpg, Popularity, Make_Audi, Make_BMW, Make_Buick, Make_Cadillac, Make_Chevrolet, Make_Chrysler, Make_Dodge, Make_Ford, Make_GMC, Make_Honda, Make_Hyundai, Make_Infiniti, Make_Kia, Make_Land Rover, Make_Lexus, Make_Lincoln, Make_Mazda, Make_Mercedes-Benz, Make_Mitsubishi, Make_Nissan, Make_Oldsmobile, Make_Pontiac, Make_Porsche, Make_Subaru, Make_Suzuki, Make_Toyota, Make_Volkswagen, Make_Volvo, Make_infrequent_sklearn, Model_911, Model_A3, Model_Accord, Model_Beetle, Model_Beetle Convertible, Model_C/K 1500 Series, Model_CTS, Model_Canyon, Model_Civic, Model_Colorado, Model_Corvette, Model_Dakota, Model_F-150, Model_F-250, Model_Frontier, Model_Pilot, Model_Ram Pickup 1500, Model_Sienna, Model_Sierra 1500, Model_Sierra 1500 Classic, Model_Silverado 1500, Model_Silverado 1500 Classic, Model_Tacoma, Model_Transit Wagon, Model_Tundra, Model_XC60, Model_infrequent_sklearn, Engine Fuel Type_diesel, Engine Fuel Type_electric, Engine Fuel Type_flex-fuel (premium unleaded recommended/E85), Engine Fuel Type_flex-fuel (premium unleaded required/E85), Engine Fuel Type_flex-fuel (unleaded/E85), Engine Fuel Type_flex-fuel (unleaded/natural gas), Engine Fuel Type_natural gas, Engine Fuel Type_premium unleaded (recommended), Engine Fuel Type_premium unleaded (required), Transmission Type_AUTOMATIC, Transmission Type_DIRECT_DRIVE, Transmission Type_MANUAL, Transmission Type_UNKNOWN, Driven_Wheels_four wheel drive, Driven_Wheels_front wheel drive, Driven_Wheels_rear wheel drive, Market Category_Crossover,Flex Fuel, Market Category_Crossover,Hatchback, Market Category_Crossover,Hybrid, Market Category_Crossover,Luxury, Market Category_Crossover,Luxury,Performance, Market Category_Crossover,Performance, Market Category_Diesel, Market Category_Diesel,Luxury, Market Category_Exotic,Factory Tuner,Luxury,High-Performance, Market Category_Exotic,High-Performance, Market Category_Exotic,Luxury,High-Performance, Market Category_Factory Tuner,High-Performance, Market Category_Factory Tuner,Luxury,High-Performance, Market Category_Factory Tuner,Performance, Market Category_Flex Fuel, Market Category_Flex Fuel,Performance, Market Category_Hatchback, Market Category_Hatchback,Hybrid, Market Category_Hatchback,Luxury, Market Category_Hatchback,Performance, Market Category_High-Performance, Market Category_Hybrid, Market Category_Luxury, Market Category_Luxury,High-Performance, Market Category_Luxury,Hybrid, Market Category_Luxury,Performance, Market Category_Performance, Market Category___缺失__, Market Category_infrequent_sklearn, Vehicle Size_Large, Vehicle Size_Midsize, Vehicle Style_2dr SUV, Vehicle Style_4dr Hatchback, Vehicle Style_4dr SUV, Vehicle Style_Cargo Minivan, Vehicle Style_Cargo Van, Vehicle Style_Convertible, Vehicle Style_Convertible SUV, Vehicle Style_Coupe, Vehicle Style_Crew Cab Pickup, Vehicle Style_Extended Cab Pickup, Vehicle Style_Passenger Minivan, Vehicle Style_Passenger Van, Vehicle Style_Regular Cab Pickup, Vehicle Style_Sedan, Vehicle Style_Wagon

## 5. 模型評估結果

| 特徵選擇方法 | 特徵數量 | MAE | MSE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|---|---|
| LassoCV (L1 正則化) | 124 | 14962.7065 | 2238732675.8440 | 47315.2478 | 0.5379 | 0.5108 |
| Sequential Feature Selection | 10 | 16829.9726 | 2494890271.8892 | 49948.8766 | 0.4851 | 0.4828 |
| Pearson 相關係數 | 10 | 17677.7673 | 2536000041.9219 | 50358.7137 | 0.4766 | 0.4742 |
| SelectKBest (f_regression) | 10 | 17677.7673 | 2536000041.9219 | 50358.7137 | 0.4766 | 0.4742 |
| RFE (LinearRegression) | 10 | 17865.1280 | 2536304299.8357 | 50361.7345 | 0.4765 | 0.4742 |

注意：Adjusted R² 公式為 Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)，其中 n 為測試集樣本數，p 為最終模型實際使用的轉換後特徵數（不含截距）。
R² 描述模型在測試集上解釋變異的比例；Adjusted R² 會依有效預測變數數量進行懲罰。

最終模型（RFE (LinearRegression)）的指標：

- MAE：17865.1280
- MSE：2536304299.8357
- RMSE：50361.7345
- R²：0.4765
- Adjusted R²：0.4742（Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)，n 為測試集樣本數，p 為最終模型實際使用的轉換後特徵數（不含截距）。）

## 6. 迴歸假設檢驗

| 假設 | 檢查方法 | 結果 | 判斷 | 說明 |
|---|---|---|---|---|
| 1. 線性關係 | Rainbow 檢定 + 殘差對擬合值圖 | F=0.6956，p=1 | 未發現足夠證據拒絕線性假設 | Rainbow 檢定的虛無假設為線性關係成立；仍需檢視殘差圖是否有彎曲趨勢。 |
| 2. 誤差獨立性 | Durbin-Watson | DW=2.0030 | 未發現明顯一階自相關證據 | DW 接近 2 代表較少的一階自相關證據（1.5～2.5 為常用經驗範圍）。本檢定主要針對順序或時間相關的殘差；橫斷面資料不可僅憑此檢定就宣稱誤差完全獨立。 |
| 3. 誤差等變異性 | Breusch-Pagan + 尺度位置圖 | LM=252.4827，p=1.631e-48 | 有證據顯示殘差變異數可能不固定 | 單一檢定不能完全排除或證明異質變異，需搭配殘差圖一起判讀。 |
| 4. 殘差常態性 | Shapiro-Wilk + 直方圖 + Q-Q Plot | W=0.4143，p=1.318e-83 | 有足夠證據拒絕殘差常態性 | 殘差樣本數超過 5000，Shapiro-Wilk 改以隨機抽樣 5000 筆計算；樣本很大時此檢定對輕微偏離非常敏感，請同時觀察 Q-Q Plot。 |
| 5. 無嚴重多重共線性 | Variance Inflation Factor (VIF) | 最大 VIF=1.6726 | 未見 VIF 超過 5 的特徵 | VIF > 5 是常見的經驗提示門檻，並非絕對標準；類別變數的虛擬變數若完全共線會造成無限大 VIF。 |

統計檢定的 p-value 不是「假設成立的機率」。未拒絕虛無假設只代表未發現足夠證據拒絕該假設，不代表假設已被證明成立。Shapiro-Wilk 對大樣本的輕微偏離很敏感，請同時參考 Q-Q Plot。

## 7. 預測圖與 95% 預測區間

預測區間使用 statsmodels OLS 的 get_prediction().summary_frame()，描述個別新觀測值的不確定性，通常比平均反應的信賴區間寬。

OLS 與 Sklearn 預測值的最大絕對差異：0.00000000

## 8. CRISP-DM 六階段

- 1. 商業理解：研究汽車規格與售價之間的關係，或使用者自訂的預測目標。
- 2. 資料理解：檢查資料筆數、欄位型態、缺失值、重複資料與描述性統計。
- 3. 資料準備：清理資料、前處理、特徵選擇與訓練／測試切分。
- 4. 建模：以 LinearRegression 建立多元線性迴歸模型，並比較五種特徵選擇方法。
- 5. 評估：比較 MAE、MSE、RMSE、R²、Adjusted R²，並進行假設檢驗與預測區間分析。
- 6. 部署：以 Streamlit 展示，說明預測結果不代表實際成交價格，且僅適用於與訓練資料相近的情境。

## 9. GPT 輔助內容

- Python 程式架構規劃與模組拆分（src/ 下各模組與 app.py 的分工）。
- 多元線性迴歸 Pipeline 實作（前處理、特徵選擇與 LinearRegression 串接）。
- 五種特徵選擇方法的程式實作（Pearson、SelectKBest、RFE、SFS、LassoCV）。
- MAE、MSE、RMSE、R² 與 Adjusted R² 的計算函式。
- 迴歸假設檢驗的設計（Rainbow、Durbin-Watson、Breusch-Pagan、Shapiro-Wilk、VIF）。
- 95% 預測區間的 statsmodels OLS 實作與 Sklearn 結果交叉檢查。
- 視覺化與分析報告的架構設計。
- CRISP-DM 六階段的內容整理。
- Streamlit 網站介面與 Streamlit Community Cloud 部署說明。

本專案部分程式與文字由 AI 助理協助產生，並經作者檢查與確認。

## 10. NotebookLM 摘要

本專案以 11199 筆汽車資料進行多元線性迴歸分析，目標變數為建議售價 MSRP。移除重複資料並丟棄目標變數缺失的列後，以 8959 筆作為訓練集、2240 筆作為測試集；數值特徵以中位數填補並標準化，類別特徵經 One-Hot Encoding 後，轉換後特徵數為 128。本次比較五種特徵選擇方法。LassoCV 保留 124 個特徵，測試集 R² 為 0.5379、RMSE 為 47315.2478，表現最佳。Sequential Feature Selection 選出 10 個特徵，R² 為 0.4851、RMSE 為 49948.8766。Pearson 相關係數與 SelectKBest 的選擇結果相同，R² 皆為 0.4766、RMSE 為 50358.7137。RFE (LinearRegression) 同樣選出 10 個特徵，R² 為 0.4765、RMSE 為 50361.7345，Adjusted R² 為 0.4742。本平台預設採用 RFE，因其特徵數較少、係數較易解讀，代價是測試集 R² 比 LassoCV 低約 0.06。最終模型的 MAE 為 17865.1280，表示預測值與實際售價的平均絕對誤差約為 17865（與 MSRP 相同單位）。假設檢驗方面：Rainbow 檢定 F=0.6956、p=1，未發現足夠證據拒絕線性假設；Durbin-Watson 為 2.0030，未見明顯一階自相關；Breusch-Pagan 的 LM=252.4827、p=1.631e-48，顯示殘差變異數可能不固定；Shapiro-Wilk 的 W=0.4143、p=1.318e-83（以隨機抽樣 5000 筆計算），殘差不符合常態分布；最大 VIF 為 1.6726，未見嚴重多重共線性。因此點估計可作為參考，而 95% 預測區間在異質變異與非常態殘差下只能視為近似。OLS 與 Sklearn 的預測值最大絕對差異為 0.00000000，兩者結果一致。本模型只適用於與訓練資料相近的情境，無法解釋因果關係，預測值也不等同實際成交價格。後續可加入交互作用項、比較非線性模型，或對售價取對數以改善殘差分布。

參考來源：

- Car Price Dataset（CooperUnion）（資料集頁面）：https://www.kaggle.com/datasets/CooperUnion/cardataset
- scikit-learn：Feature selection（統計文件）：https://scikit-learn.org/stable/modules/feature_selection.html
- statsmodels：Linear Regression Models（統計文件）：https://www.statsmodels.org/stable/regression.html
- ChatGPT 對話記錄（對話紀錄）：[data/ChatGPT Dialogue Content.pdf](data/ChatGPT%20Dialogue%20Content.pdf)

## 11. 研究限制與結論

- 線性模型只能捕捉線性關係，無法自動處理複雜的非線性效果。
- 相關性與係數不代表因果關係。
- 模型只適用於與訓練資料特性相近的資料，預測結果不代表實際成交價格。
- Durbin-Watson 主要檢查特定形式的殘差自相關；橫斷面資料不能僅憑此檢定宣稱誤差獨立。
- 測試集樣本數有限時，各指標與方法排名可能受抽樣影響。
