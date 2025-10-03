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
    'discount', 
    # 'extra_cancellation', # dont like this variable
    'quoteCreateHour', # could be categorical
    'leadTimeDays',
    'tripLengthDays',
    'numTravellers',
    'babyCount',
    'childCount',
    'seniorCount',
    'pricePerDay',
    'pricePerTraveller',
    'pricePerTravellerPerDay',
    'numBoostersApplied' 
]          

CATEGORICAL_FEATURES: List[str] = [
    'platform',
    'quoteCreateDay', 
    'isAfrica',
    'isAsia',
    'isEurope',
    'isNorthAmerica',
    'isSouthAmerica',
    'isOceania',
    'isAntarctica',
    'quoteCreateDaypart',
    # 'hasBaby', # include these three if drop them as numerical
    # 'hasChild',
    # 'hasSenior',
    'isWeekendTrip',
    'isSoloTraveller',
    'boost_Extra_Cancellation',
    'boost_Cruise_Cover',
    'boost_Snow_Sports',
    'boost_Existing_Medical_Conditions',
    'boost_Gadget_Cover',
    'boost_Motorcycle_Cover',
    'boost_Rental_Vehicle_Insurance_Excess',
    'boost_Adventure_Activities',
    'boost_Specified_Items'
]    


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
    'discount',
    # 'extra_cancellation',
    'quoteCreateHour',
    'leadTimeDays',
    'tripLengthDays',
    'numTravellers',
    'babyCount',
    'childCount',
    'seniorCount',
    'pricePerDay',
    'pricePerTraveller',
    'pricePerTravellerPerDay',
    'numBoostersApplied'
]

CATEGORICAL_FEATURES: List[str] = [
    'platform',
    'quoteCreateDay',
    'isAfrica',
    'isAsia',
    'isEurope',
    'isNorthAmerica',
    'isSouthAmerica',
    'isOceania',
    'isAntarctica',
    'quoteCreateDaypart',
    # 'hasBaby',
    # 'hasChild',
    # 'hasSenior',
    'isWeekendTrip',
    'isSoloTraveller',
    'boost_Extra_Cancellation',
    'boost_Cruise_Cover',
    'boost_Snow_Sports',
    'boost_Existing_Medical_Conditions',
    'boost_Gadget_Cover',
    'boost_Motorcycle_Cover',
    'boost_Rental_Vehicle_Insurance_Excess',
    'boost_Adventure_Activities',
    'boost_Specified_Items'
]

# ============================== Helper Functions ==============================
def validate_feature_lists(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str
) -> Tuple[List[str], List[str]]:
    if not numeric_cols and not categorical_cols:
        raise ValueError(
            "Please set NUMERIC_FEATURES and/or CATEGORICAL_FEATURES. "
            "No inference is performed in this version."
        )

    for lst_name, cols in [("NUMERIC_FEATURES", numeric_cols), ("CATEGORICAL_FEATURES", categorical_cols)]:
        if target_col in cols:
            raise ValueError(f"{lst_name} contains the target column '{target_col}', which is not allowed.")

    overlap = set(numeric_cols).intersection(set(categorical_cols))
    if overlap:
        raise ValueError(f"Columns listed in BOTH numeric and categorical: {sorted(overlap)}")

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


def _is_binary_one_hot(s: pd.Series) -> bool:
    """True if all non-null values are in {0,1,True,False}."""
    vals = set(pd.unique(s.dropna()))
    return vals.issubset({0, 1, True, False})


def split_categoricals_by_cardinality(
    train_df: pd.DataFrame, categorical_cols: List[str], threshold: int
) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    """
    Return (ohe_cols, ord_cols, binary_passthrough_cols, cardinalities_for_non_binary).
    Cardinality measured on TRAIN only.
    """
    binary_passthrough: List[str] = [c for c in categorical_cols if _is_binary_one_hot(train_df[c])]
    non_binary = [c for c in categorical_cols if c not in binary_passthrough]

    card: Dict[str, int] = {c: train_df[c].nunique(dropna=True) for c in non_binary}
    ohe_cols: List[str] = [c for c in non_binary if card[c] <= threshold]
    ord_cols: List[str] = [c for c in non_binary if card[c] > threshold]
    return ohe_cols, ord_cols, binary_passthrough, card


def build_preprocessor(
    numeric_cols: List[str],
    binary_passthrough_cols: List[str],
    ohe_cols: List[str],
    ord_cols: List[str]
) -> ColumnTransformer:
    numeric_pipe: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    # One-hot encoder: drop='first' for identifiable baseline
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        # Older scikit-learn uses `sparse` instead of `sparse_output`
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

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
    if binary_passthrough_cols:
        transformers.append(("binary", "passthrough", binary_passthrough_cols))
    if ohe_cols:
        transformers.append(("ohe", ohe_pipe, ohe_cols))
    if ord_cols:
        transformers.append(("ord", ord_pipe, ord_cols))

    pre: ColumnTransformer = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=True
    )
    return pre


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())

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
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D.")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature name length mismatch: {X.shape[1]} vs {len(feature_names)}.")
    return pd.DataFrame(X, columns=feature_names)

# ---------- Encoded-level inspectors (use after fit) ----------
def get_ohe_level_info(preprocessor: ColumnTransformer, ohe_cols: List[str]) -> Dict[str, Dict[str, List]]:
    """
    For each OHE column, return {'kept': [levels kept as dummies], 'dropped': level_dropped}.
    """
    info: Dict[str, Dict[str, List]] = {}
    if not ohe_cols or "ohe" not in preprocessor.named_transformers_:
        return info

    pipe = preprocessor.named_transformers_["ohe"]  # Pipeline
    enc: OneHotEncoder = pipe.named_steps["ohe"]
    drop = getattr(enc, "drop", None)
    drop_idx = getattr(enc, "drop_idx_", None)

    for i, col in enumerate(ohe_cols):
        cats = list(enc.categories_[i])
        if drop is None and drop_idx is None:
            kept = cats
            dropped = None
        else:
            # default for drop='first' is index 0
            di = 0
            if isinstance(drop_idx, (list, np.ndarray)) and len(drop_idx) > i and drop_idx[i] is not None:
                di = int(drop_idx[i])
            kept = [c for j, c in enumerate(cats) if j != di]
            dropped = cats[di]
        info[col] = {"kept": kept, "dropped": dropped}
    return info


def get_ord_level_info(preprocessor: ColumnTransformer, ord_cols: List[str]) -> Dict[str, Dict]:
    """
    For each Ordinal column, return mapping like {col: {'categories': [...], 'codes': range(n)}}.
    """
    info: Dict[str, Dict] = {}
    if not ord_cols or "ord" not in preprocessor.named_transformers_:
        return info

    pipe = preprocessor.named_transformers_["ord"]  # Pipeline
    enc: OrdinalEncoder = pipe.named_steps["ord"]
    for i, col in enumerate(ord_cols):
        cats = list(enc.categories_[i])
        info[col] = {"categories": cats, "code_for": {cat: code for code, cat in enumerate(cats)}}
    return info

# ============================== Build everything on import =====================
data: pd.DataFrame = pd.read_csv("cleaned_data.csv")

train_df, temp_df = train_test_split(
    data, test_size=0.30, random_state=RANDOM_STATE, stratify=data[TARGET_COL]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df[TARGET_COL]
)

# Target / Feature split
X_train_raw: pd.DataFrame = train_df.drop(columns=[TARGET_COL])
y_train: pd.Series = train_df[TARGET_COL].copy()

X_val_raw: pd.DataFrame = val_df.drop(columns=[TARGET_COL])
y_val: pd.Series = val_df[TARGET_COL].copy()

X_test_raw: pd.DataFrame = test_df.drop(columns=[TARGET_COL])
y_test: pd.Series = test_df[TARGET_COL].copy()

# Validate lists
NUMERIC_FEATURES, CATEGORICAL_FEATURES = validate_feature_lists(
    df=train_df,
    numeric_cols=NUMERIC_FEATURES,
    categorical_cols=CATEGORICAL_FEATURES,
    target_col=TARGET_COL
)

# Split categoricals (with binary passthrough)
ohe_cols, ord_cols, binary_cols, cardinalities = split_categoricals_by_cardinality(
    train_df=train_df,
    categorical_cols=CATEGORICAL_FEATURES,
    threshold=OHE_THRESHOLD
)

# Build, fit, transform
preprocessor: ColumnTransformer = build_preprocessor(
    numeric_cols=NUMERIC_FEATURES,
    binary_passthrough_cols=binary_cols,
    ohe_cols=ohe_cols,
    ord_cols=ord_cols,
)

X_train_arr: np.ndarray = preprocessor.fit_transform(X_train_raw)
X_val_arr: np.ndarray = preprocessor.transform(X_val_raw)
X_test_arr: np.ndarray = preprocessor.transform(X_test_raw)

# Names & DataFrames exposed to importers
feature_names: List[str] = transformed_feature_names(preprocessor)

X_train_proc: pd.DataFrame = to_dataframe(X_train_arr, feature_names)
X_val_proc: pd.DataFrame = to_dataframe(X_val_arr, feature_names)
X_test_proc: pd.DataFrame = to_dataframe(X_test_arr, feature_names)

# Also expose level info for external inspection (e.g., in notebooks)
ohe_level_info: Dict[str, Dict[str, List]] = get_ohe_level_info(preprocessor, ohe_cols)
ord_level_info: Dict[str, Dict] = get_ord_level_info(preprocessor, ord_cols)

# ============================== Only print when run as a script ===============
if __name__ == "__main__":
    print(f"Total records: {len(data)}")
    print(f"Train set:      {len(train_df)} ({len(train_df) / len(data) * 100:.1f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(data) * 100:.1f}%)")
    print(f"Test set:       {len(test_df)} ({len(test_df) / len(data) * 100:.1f}%)\n")

    print("=== Column Assignment Summary ===")
    print(f"[Numeric → Standardize] ({len(NUMERIC_FEATURES)}): {sorted(NUMERIC_FEATURES)}\n")

    if binary_cols:
        print(f"[Already One-Hot (passthrough)] ({len(binary_cols)}): {sorted(binary_cols)}")
    else:
        print("[Already One-Hot (passthrough)] None")

    print()
    if ohe_cols:
        print(f"[Categorical → One-Hot (drop='first', ≤ {OHE_THRESHOLD})] ({len(ohe_cols)}): {sorted(ohe_cols)}")
        print("  Cardinalities:", {c: cardinalities[c] for c in sorted(ohe_cols)})
        print("  One-Hot levels (per feature):")
        for c in sorted(ohe_cols):
            info = ohe_level_info.get(c, {})
            kept = info.get("kept", [])
            dropped = info.get("dropped", None)
            print(f"    - {c}: kept={kept}" + (f", dropped='{dropped}'" if dropped is not None else ""))
    else:
        print(f"[Categorical → One-Hot (≤ {OHE_THRESHOLD})] None")

    print()
    if ord_cols:
        print(f"[Categorical → Ordinal (> {OHE_THRESHOLD})] ({len(ord_cols)}): {sorted(ord_cols)}")
        print("  Cardinalities:", {c: cardinalities[c] for c in sorted(ord_cols)})
        print("  Ordinal mappings (category → code):")
        for c in sorted(ord_cols):
            info = ord_level_info.get(c, {})
            mapping = info.get("code_for", {})
            print(f"    - {c}: {mapping}")
    else:
        print(f"[Categorical → Ordinal (> {OHE_THRESHOLD})] None")

    print("\n=== Transformed Shapes ===")
    print(f"X_train_proc: {X_train_proc.shape}")
    print(f"X_val_proc:   {X_val_proc.shape}")
    print(f"X_test_proc:  {X_test_proc.shape}")

    print("\nSample of transformed training features:")
    print(X_train_proc.head(5))
