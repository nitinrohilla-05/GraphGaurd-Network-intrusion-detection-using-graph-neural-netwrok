"""Model registry for versioning, metadata logging, training curves, and artifact persistence."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from graphguard.detect.egraphsage import EGraphSAGEDetector, EGraphSAGENet
from graphguard.graph.features import NetFlowFeatureExtractor

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """Manages storage, semantic versioning, and loading of trained detection models."""

    def __init__(self, base_dir: str = "models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_version(self, model_name: str) -> str:
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return "v0.1.0"

        versions = []
        for p in model_dir.iterdir():
            if p.is_dir() and re.match(r"^v\d+\.\d+\.\d+$", p.name):
                parts = p.name[1:].split(".")
                versions.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if not versions:
            return "v0.1.0"

        versions.sort()
        major, minor, patch = versions[-1]
        return f"v{major}.{minor}.{patch + 1}"

    def save_model(
        self,
        model_name: str,
        model: EGraphSAGENet,
        feature_extractor: NetFlowFeatureExtractor,
        classes: List[str],
        calibrated_tau: float,
        energy_threshold: float,
        metrics: Dict[str, Any],
        training_curves: Dict[str, List[float]],
        hyperparameters: Dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """Save complete versioned model artifact package to registry."""
        ver = version or self._get_next_version(model_name)
        save_path = self.base_dir / model_name / ver
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. PyTorch weights
        torch.save(model.state_dict(), save_path / "model.pt")

        # 2. Feature extractor
        feature_extractor.save(str(save_path / "feature_extractor.joblib"))

        # 3. Metadata
        metadata = {
            "model_name": model_name,
            "version": ver,
            "saved_at": _utc_now_iso(),
            "edge_dim": model.edge_dim,
            "num_classes": len(classes),
            "classes": classes,
            "calibrated_tau": calibrated_tau,
            "energy_threshold": energy_threshold,
            "hyperparameters": hyperparameters,
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 4. Evaluation metrics
        with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 5. Training curves per Decision 23
        with open(save_path / "training_curves.json", "w", encoding="utf-8") as f:
            json.dump(training_curves, f, indent=2)

        logger.info(f"Successfully registered model '{model_name}' version '{ver}' at {save_path}")
        return ver

    def load_model(
        self,
        model_name: str,
        version: str = "latest",
        device: str = "cpu",
    ) -> Tuple[EGraphSAGEDetector, NetFlowFeatureExtractor, Dict[str, Any]]:
        """Load registered model and feature extractor."""
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in registry.")

        if version == "latest":
            versions = [
                p.name
                for p in model_dir.iterdir()
                if p.is_dir() and re.match(r"^v\d+\.\d+\.\d+$", p.name)
            ]
            if not versions:
                raise FileNotFoundError(f"No versions found for model '{model_name}'.")
            versions.sort(key=lambda v: [int(x) for x in v[1:].split(".")])
            target_version = versions[-1]
        else:
            target_version = version

        target_dir = model_dir / target_version
        if not target_dir.exists():
            raise FileNotFoundError(
                f"Version '{target_version}' not found for model '{model_name}'."
            )

        # Load metadata
        with open(target_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load feature extractor
        feature_extractor = NetFlowFeatureExtractor.load(
            str(target_dir / "feature_extractor.joblib")
        )

        # Instantiate model architecture
        hp = metadata.get("hyperparameters", {})
        net = EGraphSAGENet(
            edge_dim=metadata["edge_dim"],
            num_classes=metadata["num_classes"],
            node_dim=1,
            hidden_dim=hp.get("hidden_dim", 64),
            dropout=hp.get("dropout", 0.1),
        )
        net.load_state_dict(torch.load(target_dir / "model.pt", map_location=device))

        detector = EGraphSAGEDetector(
            model=net,
            classes=metadata["classes"],
            tau=metadata.get("calibrated_tau", 0.50),
            energy_threshold=metadata.get("energy_threshold", -5.0),
            mc_dropout=hp.get("mc_dropout", False),
            device=device,
        )

        return detector, feature_extractor, metadata
