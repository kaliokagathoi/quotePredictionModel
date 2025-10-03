from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================== Configuration =================================
TARGET_COL: str = "convert"
RANDOM_STATE: int = 42
OHE_THRESHOLD: int = 10  # <= 10 → One-Hot; > 10 → Ordinal

NUMERIC_FEATURES: List[str] = [
    'quote_price',
    'discount'
]          # TODO: fill with your numeric columns
CATEGORICAL_FEATURES: List[str] = [
    'destinations',
    'platform'
]      # TODO: fill with your categorical columns


# ============================== Helper Functions ==============================
def validate_feature_lists(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str
) -> Tuple[List[str], List[str]]:
    """
    Validate that feature lists are non-empty, exist in the DataFrame, do not overlap,
    and do not include the target column.

    :param df: Input DataFrame.
    :param numeric_cols: User-provided numeric feature names.
    :param categorical_cols: User-provided categorical feature names.
    :param target_col: Target column name.
    :return: (numeric_cols, categorical_cols) after validation.
    """
    if not numeric_cols and not categorical_cols:
        raise ValueError(
            "Please set NUMERIC_FEATURES and/or CATEGORICAL_FEATURES. "
            "No inference is performed in this version."
        )

    # Ensure target is not listed
    for lst_name, cols in [("NUMERIC_FEATURES", numeric_cols), ("CATEGORICAL_FEATURES", categorical_cols)]:
        if target_col in cols:
            raise ValueError(f"{lst_name} contains the target column '{target_col}', which is not allowed.")

    # Overlap check
    overlap = set(numeric_cols).intersection(set(categorical_cols))
    if overlap:
        raise ValueError(f"Columns listed in BOTH numeric and categorical: {sorted(overlap)}")

    # Existence check
    all_cols = set(df.columns)
    missing_num = [c for c in numeric_cols if c not in all_cols]
    missing_cat = [c for c in categorical_cols if c not in all_cols]
    msgs: List[str] = []
    if missing_num:
        msgs.append(f"Missing numeric columns: {missing_num}")
    if missing_cat:
        msgs.append(f"Missing categorical columns: {missing_cat}")
    if msgs:
        raise ValueError(" | ".join(msgs))

    return numeric_cols, categorical_cols


def split_categoricals_by_cardinality(
    train_df: pd.DataFrame, categorical_cols: List[str], threshold: int
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Split categorical columns by cardinality measured on the TRAIN split only.

    Rule: columns with <= threshold unique values → One-Hot; > threshold → Ordinal.

    :param train_df: Training DataFrame (used for cardinality to avoid leakage).
    :param categorical_cols: Categorical columns to evaluate.
    :param threshold: Integer cutoff (inclusive for OHE).
    :return: (ohe_cols, ord_cols, cardinalities)
    """
    card: Dict[str, int] = {c: train_df[c].nunique(dropna=True) for c in categorical_cols}
    ohe_cols: List[str] = [c for c in categorical_cols if card[c] <= threshold]
    ord_cols: List[str] = [c for c in categorical_cols if card[c] > threshold]
    return ohe_cols, ord_cols, card


def build_preprocessor(
    numeric_cols: List[str], ohe_cols: List[str], ord_cols: List[str]
) -> ColumnTransformer:
    """
    Create a ColumnTransformer:
      - Numeric: median impute → standardize (z-score)
      - OHE: most_frequent impute → OneHotEncoder(handle_unknown='ignore')
      - Ordinal: most_frequent impute → OrdinalEncoder(unknown_value=-1)

    :param numeric_cols: Numeric feature names.
    :param ohe_cols: Categorical columns designated for One-Hot.
    :param ord_cols: Categorical columns designated for Ordinal.
    :return: Unfitted ColumnTransformer (fit on train only).
    """
    numeric_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Back-compat for older scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ohe_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    ord_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if ohe_cols:
        transformers.append(("ohe", ohe_pipe, ohe_cols))
    if ord_cols:
        transformers.append(("ord", ord_pipe, ord_cols))

    pre: ColumnTransformer = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=True
    )
    return pre


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve feature names from a FITTED ColumnTransformer, including expanded OHE names.

    :param preprocessor: A fitted ColumnTransformer.
    :return: List of output feature names.
    """
    # Preferred path
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

    # Fallback (less descriptive)
    names: List[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                part = list(trans.get_feature_names_out(cols))
            except Exception:
                part = [f"{name}__{c}" for c in cols]
        else:
            part = [f"{name}__{c}" for c in cols]
        names.extend(part)
    return names


def to_dataframe(X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Convert a 2D numpy array to a pandas DataFrame with explicit column names.

    :param X: 2D numpy array.
    :param feature_names: Names aligned with X's columns.
    :return: DataFrame with those names.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D.")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature name length mismatch: {X.shape[1]} vs {len(feature_names)}.")
    return pd.DataFrame(X, columns=feature_names)


# ============================== Load & Split ==================================
data: pd.DataFrame = pd.read_excel("Freely_quote_data.xlsx", sheet_name="Quotes")

train_df, temp_df = train_test_split(
    data, test_size=0.30, random_state=RANDOM_STATE, stratify=data[TARGET_COL]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df[TARGET_COL]
)

print(f"Total records: {len(data)}")
print(f"Train set:      {len(train_df)} ({len(train_df) / len(data) * 100:.1f}%)")
print(f"Validation set: {len(val_df)} ({len(val_df) / len(data) * 100:.1f}%)")
print(f"Test set:       {len(test_df)} ({len(test_df) / len(data) * 100:.1f}%)\n")

# ============================== Target / Feature Split ========================
X_train_raw: pd.DataFrame = train_df.drop(columns=[TARGET_COL])
y_train: pd.Series = train_df[TARGET_COL].copy()

X_val_raw: pd.DataFrame = val_df.drop(columns=[TARGET_COL])
y_val: pd.Series = val_df[TARGET_COL].copy()

X_test_raw: pd.DataFrame = test_df.drop(columns=[TARGET_COL])
y_test: pd.Series = test_df[TARGET_COL].copy()

# ============================== User Feature Lists (Validated) ================
NUMERIC_FEATURES, CATEGORICAL_FEATURES = validate_feature_lists(
    df=train_df, numeric_cols=NUMERIC_FEATURES, categorical_cols=CATEGORICAL_FEATURES, target_col=TARGET_COL
)

# ============================== Cardinality-Based Split =======================
ohe_cols, ord_cols, cardinalities = split_categoricals_by_cardinality(
    train_df=train_df, categorical_cols=CATEGORICAL_FEATURES, threshold=OHE_THRESHOLD
)

# ============================== Reporting ====================================
print("=== Column Assignment Summary ===")
print(f"[Numeric → Standardize] ({len(NUMERIC_FEATURES)}): {sorted(NUMERIC_FEATURES)}\n")

if ohe_cols:
    print(f"[Categorical → One-Hot (≤ {OHE_THRESHOLD})] ({len(ohe_cols)}): {sorted(ohe_cols)}")
    print("  Cardinalities:", {c: cardinalities[c] for c in sorted(ohe_cols)})
else:
    print(f"[Categorical → One-Hot (≤ {OHE_THRESHOLD})] None")

print()

if ord_cols:
    print(f"[Categorical → Ordinal (> {OHE_THRESHOLD})] ({len(ord_cols)}): {sorted(ord_cols)}")
    print("  Cardinalities:", {c: cardinalities[c] for c in sorted(ord_cols)})
else:
    print(f"[Categorical → Ordinal (> {OHE_THRESHOLD})] None")

# ============================== Build, Fit, Transform =========================
preprocessor: ColumnTransformer = build_preprocessor(
    numeric_cols=NUMERIC_FEATURES,
    ohe_cols=ohe_cols,
    ord_cols=ord_cols,
)

X_train_arr: np.ndarray = preprocessor.fit_transform(X_train_raw)
X_val_arr: np.ndarray = preprocessor.transform(X_val_raw)
X_test_arr: np.ndarray = preprocessor.transform(X_test_raw)

# ============================== Names & DataFrames ============================
feature_names: List[str] = transformed_feature_names(preprocessor)

X_train_proc: pd.DataFrame = to_dataframe(X_train_arr, feature_names)
X_val_proc: pd.DataFrame = to_dataframe(X_val_arr, feature_names)
X_test_proc: pd.DataFrame = to_dataframe(X_test_arr, feature_names)

print("\n=== Transformed Shapes ===")
print(f"X_train_proc: {X_train_proc.shape}")
print(f"X_val_proc:   {X_val_proc.shape}")
print(f"X_test_proc:  {X_test_proc.shape}")

print("\nSample of transformed training features:")
print(X_train_proc.head(5))
