"""資料讀取模組：支援本機檔案、上傳檔案與公開 CSV 網址。

所有讀取失敗都會丟出 DataLoadError，並附上可讓使用者理解的繁體中文說明，
不會回傳看似真實的假資料。
"""

from __future__ import annotations

import io
import socket
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from . import config


class DataLoadError(Exception):
    """資料無法讀取時丟出的例外，訊息可直接顯示給使用者。"""


def decode_csv_bytes(content: bytes, source: str) -> pd.DataFrame:
    """依序嘗試多種編碼解析 CSV 位元組內容。"""
    if not content or not content.strip():
        raise DataLoadError(f"{source} 內容為空，請確認檔案或網址是否正確。")

    head = content[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise DataLoadError(
            f"{source} 回傳的是 HTML 網頁，而不是 CSV 原始檔。"
            "請改用 CSV 原始檔網址（例如 raw.githubusercontent.com 的連結），"
            "或下載 CSV 後使用「上傳 CSV」功能。"
        )

    last_error: Exception | None = None
    for encoding in config.ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"{source} 沒有可解析的欄位資料。") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"{source} 不是格式正確的 CSV：{exc}") from exc
        except Exception as exc:  # noqa: BLE001 - 轉為使用者可理解的訊息
            raise DataLoadError(f"{source} 解析失敗：{exc}") from exc

        if df.shape[1] < 2:
            raise DataLoadError(
                f"{source} 只解析出 {df.shape[1]} 個欄位，請確認分隔符號是否為逗號。"
            )
        df.columns = [str(col).strip() for col in df.columns]
        return df

    raise DataLoadError(
        f"{source} 無法以 {', '.join(config.ENCODINGS)} 任一編碼解析：{last_error}"
    )


def load_local_csv(path: Path = config.CAR_LOCAL_CSV_PATH) -> pd.DataFrame:
    """讀取本機 CSV 檔案（預設為 data/data.csv）。"""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(
            f"找不到本機檔案：{path.as_posix()}。\n"
            "請依照 data/README.md 的說明，將 Kaggle 下載的 CSV 放到 data 資料夾並命名為 data.csv；"
            "或改用「上傳 CSV」或「公開 CSV 網址」。"
        )
    return decode_csv_bytes(path.read_bytes(), source=f"本機檔案 {path.name}")


def load_csv_from_url(url: str) -> pd.DataFrame:
    """從公開 CSV 網址下載並解析資料。"""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise DataLoadError("網址格式不正確，需以 http:// 或 https:// 開頭。")

    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (multiple-linear-regression-platform)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=config.REQUEST_TIMEOUT) as response:
            content = response.read(config.MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise DataLoadError(
            f"網址回應 HTTP {exc.code}，無法取得資料。請確認此為公開可直接下載的 CSV 網址。"
            "Kaggle 資料集通常需要登入，請改用上傳 CSV。"
        ) from exc
    except urllib.error.URLError as exc:
        raise DataLoadError(f"網路連線失敗或網址無法連線：{exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise DataLoadError("連線逾時，請稍後再試或改用上傳 CSV。") from exc
    except OSError as exc:
        raise DataLoadError(f"讀取網址時發生錯誤：{exc}") from exc

    if len(content) > config.MAX_DOWNLOAD_BYTES:
        raise DataLoadError("檔案超過 100 MB 上限，請改用較小的資料集。")

    return decode_csv_bytes(content, source="公開網址")


def load_startup_50() -> pd.DataFrame:
    """讀取內建的 Startup 50 範例資料集。"""
    return load_csv_from_url(config.STARTUP_URL)


def load_uploaded_csv(content: bytes, file_name: str = "上傳檔案") -> pd.DataFrame:
    """解析使用者上傳的 CSV 位元組內容。"""
    if len(content) > config.MAX_DOWNLOAD_BYTES:
        raise DataLoadError("上傳檔案超過 100 MB 上限，請改用較小的資料集。")
    return decode_csv_bytes(content, source=f"上傳檔案 {file_name}")
