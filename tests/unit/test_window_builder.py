"""Unit tests for window builder and timestamp inspection."""

import numpy as np
import pandas as pd

from graphguard.graph.features import NetFlowFeatureExtractor
from graphguard.graph.window_builder import (
    WindowBuilder,
    build_graph_window_from_dataframe,
    inspect_timestamp_column,
)


def test_inspect_timestamp_column() -> None:
    """Detect presence or absence of flow timestamp candidate columns."""
    df_with_time = pd.DataFrame({"FLOW_START_MILLISECONDS": [1000], "SRC_IP": ["10.0.0.1"]})
    has_time, col = inspect_timestamp_column(df_with_time)
    assert has_time is True
    assert col == "FLOW_START_MILLISECONDS"

    df_no_time = pd.DataFrame({"SRC_IP": ["10.0.0.1"], "DST_IP": ["10.0.0.2"]})
    has_time, col = inspect_timestamp_column(df_no_time)
    assert has_time is False
    assert col is None


def test_build_graph_window_from_dataframe() -> None:
    """Validate GraphWindow construction from dataframe chunk."""
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["192.168.1.10", "192.168.1.11"],
            "IPV4_DST_ADDR": ["10.0.0.5", "10.0.0.5"],
            "L4_DST_PORT": [80, 443],
            "PROTOCOL": [6, 6],
            "Label": [0, 1],
            "Attack": ["Benign", "PortScan"],
        }
    )
    features = np.ones((2, 4), dtype=np.float32)

    window, labels, attack_names = build_graph_window_from_dataframe(df, features)

    assert len(window.edges) == 2
    assert len(window.node_ids) == 3  # 192.168.1.10, 192.168.1.11, 10.0.0.5
    assert window.edges[0].dst_port == 80
    assert window.edges[0].proto == "tcp"
    assert labels[0] == 0
    assert labels[1] == 1
    assert attack_names[1] == "PortScan"


def test_window_builder_chunking() -> None:
    """WindowBuilder creates expected number of chunks from consecutive rows."""
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"192.168.1.{i}" for i in range(10)],
            "IPV4_DST_ADDR": ["10.0.0.1"] * 10,
            "L4_DST_PORT": [80] * 10,
            "PROTOCOL": [6] * 10,
            "IN_BYTES": [100] * 10,
        }
    )
    extractor = NetFlowFeatureExtractor()
    extractor.fit(df)

    builder = WindowBuilder(chunk_size=4)
    windows = builder.build_windows(df, extractor)

    # 10 rows with chunk size 4 -> 3 chunks (4, 4, 2)
    assert len(windows) == 3
    assert len(windows[0][0].edges) == 4
    assert len(windows[2][0].edges) == 2


def test_window_builder_real_time_windows() -> None:
    """WindowBuilder groups flows into discrete real time windows using timestamps."""
    df = pd.DataFrame(
        {
            "FLOW_START_MILLISECONDS": [
                1421927376000,  # t = 0s -> bucket 0
                1421927390000,  # t = 14s -> bucket 0
                1421927440000,  # t = 64s -> bucket 1
                1421927500000,  # t = 124s -> bucket 2
            ],
            "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "IPV4_DST_ADDR": ["10.0.0.5", "10.0.0.5", "10.0.0.6", "10.0.0.6"],
            "L4_DST_PORT": [80, 443, 80, 443],
            "PROTOCOL": [6, 6, 6, 6],
            "IN_BYTES": [100, 200, 300, 400],
        }
    )
    extractor = NetFlowFeatureExtractor()
    extractor.fit(df)

    builder = WindowBuilder(window_duration_seconds=60.0)
    windows = builder.build_windows(df, extractor, return_dfs=True)

    # 4 flows across 3 60-second time intervals: [0, 60), [60, 120), [120, 180)
    assert len(windows) == 3
    assert len(windows[0][0].edges) == 2  # first two flows in bucket 0
    assert len(windows[1][0].edges) == 1  # third flow in bucket 1
    assert len(windows[2][0].edges) == 1  # fourth flow in bucket 2
    assert len(windows[0][3]) == 2  # return_dfs returns sub_df
