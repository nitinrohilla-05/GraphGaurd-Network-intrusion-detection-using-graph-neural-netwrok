"""Unit tests for ModelRegistry artifact persistence and versioning."""

from pathlib import Path

import pandas as pd

from graphguard.detect.egraphsage import EGraphSAGENet
from graphguard.detect.registry import ModelRegistry
from graphguard.graph.features import NetFlowFeatureExtractor


def test_registry_save_and_load(tmp_path: Path) -> None:
    """Save model to registry, verify semantic versioning, and reload."""
    registry = ModelRegistry(str(tmp_path))

    net = EGraphSAGENet(edge_dim=4, num_classes=2, hidden_dim=16)
    extractor = NetFlowFeatureExtractor()
    dummy_df = pd.DataFrame({"IN_BYTES": [10.0, 20.0], "IN_PKTS": [1.0, 2.0]})
    extractor.fit(dummy_df)

    ver1 = registry.save_model(
        model_name="test_model",
        model=net,
        feature_extractor=extractor,
        classes=["Benign", "Attack"],
        calibrated_tau=0.65,
        energy_threshold=-4.2,
        metrics={"f1": 0.95},
        training_curves={"train_loss": [0.5, 0.2]},
        hyperparameters={"hidden_dim": 16},
    )
    assert ver1 == "v0.1.0"

    # Save second version
    ver2 = registry.save_model(
        model_name="test_model",
        model=net,
        feature_extractor=extractor,
        classes=["Benign", "Attack"],
        calibrated_tau=0.70,
        energy_threshold=-4.0,
        metrics={"f1": 0.98},
        training_curves={"train_loss": [0.4, 0.1]},
        hyperparameters={"hidden_dim": 16},
    )
    assert ver2 == "v0.1.1"

    # Reload latest
    detector, loaded_extractor, metadata = registry.load_model("test_model", version="latest")
    assert detector.tau == 0.70
    assert metadata["version"] == "v0.1.1"
    assert loaded_extractor.feature_names == extractor.feature_names
