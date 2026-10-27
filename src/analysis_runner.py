"""完整分析流程：切分資料、五種特徵選擇比較、最終模型、假設檢驗與預測區間。

設計原則：
1. 所有方法共用同一份訓練／測試切分與相同的前處理規則。
2. 前處理與特徵選擇只在訓練資料上擬合。
3. 任何指標都由實際擬合的模型在測試集上計算；失敗的方法會記錄錯誤，不填入假數字。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .assumption_tests import run_assumption_tests
from .data_preprocessing import build_preprocessor, split_feature_types
from .evaluation import comparison_table, regression_metrics
from .feature_selection import (
    METHOD_DESCRIPTIONS,
    METHOD_LASSO,
    METHODS,
    selected_feature_names,
    selection_detail,
)
from .regression_model import (
    build_model_pipeline,
    count_transformed_features,
    fit_ols,
    model_coefficients,
    prediction_interval,
    selected_design_matrix,
)


def run_full_analysis(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    settings: dict,
) -> dict:
    """執行完整分析。

    Args:
        df: 已清理的資料（僅包含目標與候選特徵欄位）。
        target: 目標欄位名稱。
        features: 使用者選擇的輸入特徵欄位。
        settings: 含 test_size、random_state、cv_folds、k、direction、final_method、alpha。

    Returns:
        分析結果字典。若無法分析，會丟出 ValueError 並附上原因。
    """
    if not features:
        raise ValueError("請至少選擇一個輸入特徵。")
    if target in features:
        raise ValueError("目標變數不可同時作為輸入特徵，請移除。")

    numeric_cols, categorical_cols, skipped = split_feature_types(df, features)
    if skipped:
        features = [f for f in features if f not in skipped]
    if not features:
        raise ValueError("所選特徵皆為無法使用的欄位型態（例如日期時間），請重新選擇。")

    X = df[features].copy()
    y = df[target].astype(float)

    random_state = int(settings["random_state"])
    alpha = float(settings.get("alpha", config.ALPHA))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(settings["test_size"]), random_state=random_state
    )
    if len(X_train) < config.MIN_TRAIN_ROWS:
        raise ValueError(
            f"訓練資料只有 {len(X_train)} 筆，少於最低需求 {config.MIN_TRAIN_ROWS} 筆，無法可靠建模。"
        )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    n_transformed = count_transformed_features(preprocessor, X_train, y_train)
    cv = int(settings["cv_folds"])
    k = int(settings["k"])
    direction = settings.get("direction", "forward")

    method_results: list[dict] = []
    for method in METHODS:
        entry: dict = {
            "method": method,
            "description": METHOD_DESCRIPTIONS[method],
            "n_selected": None,
            "selected": [],
            "detail": None,
            "score_label": "",
            "notes": [],
            "metrics": None,
            "error": None,
        }
        try:
            pipeline, k_eff, notes = build_model_pipeline(
                preprocessor, method, k, direction, cv, n_transformed
            )
            pipeline.fit(X_train, y_train)
            selected = selected_feature_names(pipeline)
            detail, score_label = selection_detail(pipeline, method)
            y_pred = pipeline.predict(X_test)
            metrics = regression_metrics(y_test, y_pred, n_predictors=len(selected))
            entry.update(
                {
                    "n_selected": len(selected),
                    "selected": selected,
                    "detail": detail,
                    "score_label": score_label,
                    "notes": notes,
                    "metrics": metrics,
                    "k_used": k_eff,
                    "pipeline": pipeline,
                }
            )
            if method == METHOD_LASSO and pipeline.named_steps["selector"].used_fallback_:
                entry["notes"].append("所有 Lasso 係數皆為零，已改保留與目標相關係數絕對值最大的一個特徵。")
        except Exception as exc:  # noqa: BLE001 - 記錄失敗原因，不填入假結果
            entry["error"] = f"{type(exc).__name__}: {exc}"
        method_results.append(entry)

    valid = [m for m in method_results if m["metrics"] is not None]
    if not valid:
        errors = "；".join(f"{m['method']}: {m['error']}" for m in method_results)
        raise ValueError(f"所有特徵選擇方法都無法完成分析。{errors}")

    comparison_df = comparison_table(method_results)

    # 最終模型
    final_method = settings.get("final_method", METHODS[1])
    final_entry = next((m for m in valid if m["method"] == final_method), valid[0])
    final_pipe = final_entry["pipeline"]

    design_train = selected_design_matrix(final_pipe, X_train)
    design_test = selected_design_matrix(final_pipe, X_test)
    ols = fit_ols(design_train, y_train)
    pi = prediction_interval(ols, design_test, alpha=alpha)

    sklearn_pred = np.asarray(final_pipe.predict(X_test), dtype=float)
    ols_pred = np.asarray(pi["mean"], dtype=float)
    max_abs_diff = float(np.max(np.abs(sklearn_pred - ols_pred))) if len(sklearn_pred) else 0.0

    predictions_df = pd.DataFrame(
        {
            "實際值": y_test.to_numpy(dtype=float),
            "預測值": sklearn_pred,
            "殘差(實際-預測)": y_test.to_numpy(dtype=float) - sklearn_pred,
            "95%預測區間下界": pi["obs_ci_lower"].to_numpy(dtype=float),
            "95%預測區間上界": pi["obs_ci_upper"].to_numpy(dtype=float),
        },
        index=y_test.index,
    ).reset_index().rename(columns={"index": "測試集原始列索引"})

    final_metrics = final_entry["metrics"]
    assumptions = run_assumption_tests(ols, design_train, random_state=random_state, alpha=alpha)
    coef_df = model_coefficients(ols)

    result = {
        "target": target,
        "features_input": features,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "skipped_cols": skipped,
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_transformed_features": int(n_transformed),
        "effective_feature_warning": n_transformed < config.MIN_EFFECTIVE_FEATURES,
        "method_results": method_results,
        "comparison_table": comparison_df,
        "final_method": final_entry["method"],
        "final_pipeline": final_pipe,
        "final_selected": final_entry["selected"],
        "final_selection_detail": final_entry["detail"],
        "final_metrics": final_metrics,
        "final_notes": final_entry["notes"],
        "ols": ols,
        "ols_coefficients": coef_df,
        "prediction_interval": pi,
        "predictions": predictions_df,
        "sklearn_vs_ols_max_diff": max_abs_diff,
        "assumptions": assumptions,
        "settings": dict(settings),
        "alpha": alpha,
    }
    return result
