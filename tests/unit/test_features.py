"""Unit tests for NetFlow feature extraction and anti-leakage guards."""

import numpy as np
import pandas as pd
import pytest

from graphguard.graph.features import NetFlowFeatureExtractor


def test_feature_extractor_excludes_leakage_columns() -> None:
    """Excluded identifiers (IPs, ports, labels) must not appear in numeric features."""
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["192.168.1.1", "192.168.1.2"],
            "IPV4_DST_ADDR": ["10.0.0.1", "10.0.0.2"],
            "L4_SRC_PORT": [44321, 55432],
            "L4_DST_PORT": [80, 443],
            "IN_BYTES": [1000, 2500],
            "IN_PKTS": [10, 25],
            "OUT_BYTES": [500, 1200],
            "OUT_PKTS": [5, 12],
            "TCP_FLAGS": [24, 16],
            "Label": [0, 1],
            "Attack": ["Benign", "PortScan"],
        }
    )

    extractor = NetFlowFeatureExtractor()
    extractor.fit(df)

    assert "IPV4_SRC_ADDR" not in extractor.feature_names
    assert "IPV4_DST_ADDR" not in extractor.feature_names
    assert "L4_SRC_PORT" not in extractor.feature_names
    assert "Label" not in extractor.feature_names
    assert "Attack" not in extractor.feature_names

    # Kept features should be the numeric NetFlow fields
    assert "IN_BYTES" in extractor.feature_names
    assert "L4_DST_PORT" in extractor.feature_names


def test_feature_extractor_fit_transform_and_persistence(tmp_path: pytest.TempPathFactory) -> None:
    """Extractor should scale features, save state, and reload cleanly."""
    train_df = pd.DataFrame(
        {
            "IN_BYTES": [10.0, 1000.0, 50000.0],
            "IN_PKTS": [1.0, 10.0, 250.0],
            "FLOW_DURATION_MILLISECONDS": [5.0, 50.0, 500.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "IN_BYTES": [500.0],
            "IN_PKTS": [5.0],
            "FLOW_DURATION_MILLISECONDS": [25.0],
        }
    )

    extractor = NetFlowFeatureExtractor()
    X_train = extractor.fit_transform(train_df)
    assert X_train.shape == (3, 3)

    X_test = extractor.transform(test_df)
    assert X_test.shape == (1, 3)

    # Persistence
    save_path = tmp_path / "extractor.joblib"
    extractor.save(str(save_path))

    loaded_extractor = NetFlowFeatureExtractor.load(str(save_path))
    assert loaded_extractor.feature_names == extractor.feature_names
    X_test_reloaded = loaded_extractor.transform(test_df)
    np.testing.assert_allclose(X_test, X_test_reloaded)
