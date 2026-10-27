"""專案共用設定。所有路徑皆相對於專案根目錄，避免依賴本機絕對路徑。"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Problem 1：汽車價格預測（Kaggle CooperUnion/cardataset）
CAR_KAGGLE_PAGE = "https://www.kaggle.com/datasets/CooperUnion/cardataset"
CAR_LOCAL_CSV_PATH = DATA_DIR / "data.csv"
CAR_TARGET = "MSRP"
CAR_EXPECTED_RAW_COLUMNS = 16

# Problem 2：自訂資料集（預設 Startup 50）
STARTUP_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
STARTUP_TARGET = "Profit"

# 統計與實驗設定
ALPHA = 0.05
PI_CONFIDENCE = 0.95
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5
DEFAULT_K = 10
MIN_EFFECTIVE_FEATURES = 10
MIN_TRAIN_ROWS = 20
MAX_CATEGORIES = 30
BACKWARD_SFS_MAX_FEATURES = 30
VIF_THRESHOLD = 5.0
SHAPIRO_MAX_N = 5000
TARGET_NUMERIC_RATIO = 0.95

# 資料讀取
ENCODINGS = ("utf-8", "utf-8-sig", "big5", "cp950")
REQUEST_TIMEOUT = 30
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

# ID 欄位名稱偵測規則（不分大小寫）
ID_NAME_PATTERN = r"^(id|index|unnamed)|(^|[\s_\-])id($|[\s_\-])|identifier"
