"""Graph window constructor from NetFlow records with timestamp inspection and chunking fallback."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graphguard.core.models import EdgeKey, GraphWindow
from graphguard.graph.features import NetFlowFeatureExtractor

logger = logging.getLogger(__name__)

TIMESTAMP_CANDIDATES = [
    "flow_start_milliseconds",
    "flow_start_sec",
    "timestamp",
    "time",
    "flow_start_time",
]

SRC_IP_CANDIDATES = ["ipv4_src_addr", "source ip", "src_ip", "srcip"]
DST_IP_CANDIDATES = ["ipv4_dst_addr", "destination ip", "dst_ip", "dstip"]
DST_PORT_CANDIDATES = ["l4_dst_port", "destination port", "dst_port", "dport"]
PROTO_CANDIDATES = ["protocol", "proto"]


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def inspect_timestamp_column(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Inspect CSV columns for flow timestamp indicators."""
    col = _find_column(df, TIMESTAMP_CANDIDATES)
    return (col is not None, col)


def normalize_protocol(val: object) -> str:
    """Map numeric or text protocol identifiers to canonical string names."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        int_val = int(val)
        if int_val == 6:
            return "tcp"
        if int_val == 17:
            return "udp"
        if int_val == 1:
            return "icmp"
        return str(int_val)
    val_str = str(val).strip().lower()
    return val_str or "tcp"


def build_graph_window_from_dataframe(
    df: pd.DataFrame,
    features_matrix: np.ndarray,
    shard_id: str = "default",
    t_start: Optional[datetime] = None,
    t_end: Optional[datetime] = None,
) -> Tuple[GraphWindow, np.ndarray, np.ndarray]:
    """Convert a dataframe chunk into a GraphWindow, returning (window, labels, attack_names).

    Labels: binary int array (0 = benign, 1 = attack).
    Attack names: string array of raw attack categories.
    """
    if len(df) != len(features_matrix):
        raise ValueError("DataFrame length must match features_matrix length.")

    src_col = _find_column(df, SRC_IP_CANDIDATES)
    dst_col = _find_column(df, DST_IP_CANDIDATES)
    dport_col = _find_column(df, DST_PORT_CANDIDATES)
    proto_col = _find_column(df, PROTO_CANDIDATES)

    if not src_col or not dst_col:
        raise ValueError(
            f"Cannot construct graph window: missing IP columns. "
            f"Available columns: {list(df.columns)}"
        )

    # Label extraction
    label_col = _find_column(df, ["label"])
    attack_col = _find_column(df, ["attack"])

    if label_col:
        raw_labels = df[label_col].to_numpy()
        # Handle string ("BENIGN" / "Attack") or integer labels
        if np.issubdtype(raw_labels.dtype, np.number):
            labels = (raw_labels > 0).astype(int)
        else:
            labels = np.array(
                [0 if str(lbl).strip().upper() == "BENIGN" else 1 for lbl in raw_labels],
                dtype=int,
            )
    else:
        labels = np.zeros(len(df), dtype=int)

    if attack_col:
        attack_names = df[attack_col].astype(str).to_numpy()
    elif label_col:
        attack_names = df[label_col].astype(str).to_numpy()
    else:
        attack_names = np.array(["Benign"] * len(df))

    # Node indexing
    node_to_idx: Dict[str, int] = {}
    node_ids: List[str] = []

    def get_node_idx(ip: str) -> int:
        if ip not in node_to_idx:
            idx = len(node_ids)
            node_to_idx[ip] = idx
            node_ids.append(ip)
            return idx
        return node_to_idx[ip]

    edge_indices: List[List[int]] = []
    edges: List[EdgeKey] = []
    edge_attr_list: List[List[float]] = []

    src_ips = df[src_col].astype(str).str.strip().to_numpy()
    dst_ips = df[dst_col].astype(str).str.strip().to_numpy()
    dports = (
        pd.to_numeric(df[dport_col], errors="coerce").fillna(80).astype(int).to_numpy()
        if dport_col
        else np.full(len(df), 80, dtype=int)
    )
    protos = [normalize_protocol(p) for p in df[proto_col]] if proto_col else ["tcp"] * len(df)

    for i in range(len(df)):
        s_ip = src_ips[i]
        d_ip = dst_ips[i]
        dp = int(dports[i])
        pr = protos[i]

        u = get_node_idx(s_ip)
        v = get_node_idx(d_ip)

        edge_indices.append([u, v])
        edges.append(EdgeKey(src_ip=s_ip, dst_ip=d_ip, dst_port=dp, proto=pr))
        edge_attr_list.append(features_matrix[i].tolist())

    time_col = _find_column(df, TIMESTAMP_CANDIDATES)
    if time_col and t_start is None:
        try:
            min_ts = float(df[time_col].min())
            start_time = datetime.fromtimestamp(min_ts / 1000.0, tz=timezone.utc)
            end_col = _find_column(df, ["flow_end_milliseconds", "flow_end_sec", "flow_end_time"])
            if end_col:
                max_ts = float(df[end_col].max())
            else:
                max_ts = float(df[time_col].max())
            end_time = datetime.fromtimestamp(max_ts / 1000.0, tz=timezone.utc)
        except Exception:
            now = datetime.now(timezone.utc)
            start_time = now
            end_time = now + timedelta(seconds=30)
    else:
        now = datetime.now(timezone.utc)
        start_time = t_start or now
        end_time = t_end or (start_time + timedelta(seconds=30))

    window = GraphWindow(
        edge_index=edge_indices,
        edge_attr=edge_attr_list,
        node_ids=node_ids,
        edges=edges,
        t_start=start_time,
        t_end=end_time,
        shard_id=shard_id,
        labels=labels.tolist() if hasattr(labels, "tolist") else list(labels),
    )

    return window, labels, attack_names


class WindowBuilder:
    """Manages window generation using timestamps (real time windows) or consecutive row chunks."""

    def __init__(
        self,
        chunk_size: int = 50000,
        window_duration_seconds: Optional[float] = 60.0,
    ) -> None:
        self.chunk_size = chunk_size
        self.window_duration_seconds = window_duration_seconds

    def build_windows(
        self,
        df: pd.DataFrame,
        feature_extractor: NetFlowFeatureExtractor,
        return_dfs: bool = False,
    ) -> List[Any]:
        """Split dataframe into windows, normalize features, and return window tuples.

        If timestamps are present and window_duration_seconds is set, generates real time windows.
        Otherwise falls back to consecutive row chunks of size self.chunk_size.
        """
        has_time, time_col = inspect_timestamp_column(df)
        windows: List[Any] = []

        if has_time and time_col and self.window_duration_seconds is not None:
            logger.info(
                f"Using flow timestamp column '{time_col}' for real time windowing "
                f"(window_duration={self.window_duration_seconds}s)."
            )
            df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
            X_all = feature_extractor.transform(df_sorted)

            raw_ts = df_sorted[time_col].astype(float).to_numpy()
            divisor = 1000.0 if np.nanmax(raw_ts) > 1e11 else 1.0
            ts_sec = raw_ts / divisor
            bucket_ids = (ts_sec // self.window_duration_seconds).astype(np.int64)

            for b_id, group_df in df_sorted.groupby(bucket_ids, sort=False):
                group_indices = group_df.index.to_numpy()
                sub_X = X_all[group_indices]
                min_sec = float(ts_sec[group_indices[0]])
                max_sec = float(ts_sec[group_indices[-1]])
                t_start = datetime.fromtimestamp(min_sec, tz=timezone.utc)
                t_end = datetime.fromtimestamp(max_sec, tz=timezone.utc)
                win, lbls, atts = build_graph_window_from_dataframe(
                    group_df, sub_X, shard_id=f"win_{b_id}", t_start=t_start, t_end=t_end
                )
                win.window_id = f"win_{b_id}"
                if return_dfs:
                    windows.append((win, lbls, atts, group_df))
                else:
                    windows.append((win, lbls, atts))
        else:
            logger.info(
                f"No timestamp column found or time windowing disabled. Building windows from "
                f"consecutive row chunks (size={self.chunk_size}), assuming row order is "
                f"chronological."
            )
            X_all = feature_extractor.transform(df)
            n_rows = len(df)
            for start_idx in range(0, n_rows, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_rows)
                sub_df = df.iloc[start_idx:end_idx]
                sub_X = X_all[start_idx:end_idx]
                win, lbls, atts = build_graph_window_from_dataframe(sub_df, sub_X)
                win.window_id = f"chunk_{start_idx // self.chunk_size}"
                if return_dfs:
                    windows.append((win, lbls, atts, sub_df))
                else:
                    windows.append((win, lbls, atts))

        return windows
