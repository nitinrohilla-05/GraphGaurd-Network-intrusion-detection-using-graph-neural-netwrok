"""NetFlow numeric feature extraction and normalization with strict anti-leakage guards."""

import logging
from typing import List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical leakage columns that must never be used as edge features
DEFAULT_EXCLUDED_COLUMNS = {
    "ipv4_src_addr",
    "ipv4_dst_addr",
    "l4_src_port",
    "label",
    "attack",
    "source ip",
    "destination ip",
    "source port",
    "flow id",
    "timestamp",
    "flow_start_milliseconds",
    "flow_end_milliseconds",
}


class NetFlowFeatureExtractor:
    """Extracts, log-scales, and standardizes numeric NetFlow features.

    Scaler is fitted strictly on the training split to prevent data leakage.
    """

    def __init__(self, excluded_columns: Optional[Set[str]] = None) -> None:
        self.excluded_columns = {
            c.lower().strip() for c in (excluded_columns or DEFAULT_EXCLUDED_COLUMNS)
        }
        self.feature_names: List[str] = []
        self.log_transform_columns: List[str] = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _select_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = []
        for col in df.columns:
            clean_col = str(col).lower().strip()
            if clean_col in self.excluded_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(str(col))
        return numeric_cols

    def _determine_log_columns(self, feature_cols: List[str]) -> List[str]:
        log_keywords = ("byte", "pkt", "duration", "len", "rate", "iat", "win")
        log_cols = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in log_keywords):
                log_cols.append(col)
        return log_cols

    def fit(self, train_df: pd.DataFrame) -> "NetFlowFeatureExtractor":
        """Fit scaler on training data only."""
        self.feature_names = self._select_numeric_columns(train_df)
        self.log_transform_columns = self._determine_log_columns(self.feature_names)

        X = train_df[self.feature_names].copy()
        for col in self.log_transform_columns:
            # log1p transformation on positive values
            X[col] = np.log1p(np.clip(X[col].to_numpy(dtype=float), a_min=0.0, a_max=None))

        # Replace any residual NaNs or Infs
        X = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info(
            f"Fitted NetFlowFeatureExtractor on {len(train_df)} rows. "
            f"Selected {len(self.feature_names)} features "
            f"({len(self.log_transform_columns)} log-scaled)."
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into normalized feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("NetFlowFeatureExtractor must be fitted before transforming.")

        X = df[self.feature_names].copy()
        for col in self.log_transform_columns:
            X[col] = np.log1p(np.clip(X[col].to_numpy(dtype=float), a_min=0.0, a_max=None))

        X_mat = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        return self.scaler.transform(X_mat).astype(np.float32)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: str) -> None:
        """Serialize extractor state to disk."""
        state = {
            "feature_names": self.feature_names,
            "log_transform_columns": self.log_transform_columns,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "excluded_columns": list(self.excluded_columns),
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str) -> "NetFlowFeatureExtractor":
        """Load extractor state from disk."""
        state = joblib.load(filepath)
        instance = cls(excluded_columns=set(state.get("excluded_columns", [])))
        instance.feature_names = state["feature_names"]
        instance.log_transform_columns = state["log_transform_columns"]
        instance.scaler = state["scaler"]
        instance.is_fitted = state["is_fitted"]
        return instance
