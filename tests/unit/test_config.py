"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from graphguard.core.config import load_config


def test_load_default_config() -> None:
    """Default config must load and populate default settings."""
    cfg = load_config(config_path="config/default.yaml")
    assert cfg.system.env in ["development", "production", "test"]
    assert cfg.detection.threshold_tau == 0.70
    assert cfg.containment.k_hops == 2
    assert cfg.containment.adam_steps == 200
    assert cfg.bus.bootstrap_servers == "localhost:9092"
    assert "flows_raw" in cfg.bus.topics
    assert cfg.default_enforcer == "DryRunEnforcer"


def test_assets_yaml_schema() -> None:
    """Validate structure of config/assets.yaml."""
    assets_file = Path("config/assets.yaml")
    assert assets_file.exists()
    with open(assets_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "cost_weights" in data
    assert "alpha" in data["cost_weights"]
    assert "beta" in data["cost_weights"]
    assert "roles" in data
    assert "gateway" in data["roles"]
    assert data["roles"]["gateway"]["criticality"] == 1.0


def test_action_policy_yaml_schema() -> None:
    """Validate structure of config/action_policy.yaml."""
    policy_file = Path("config/action_policy.yaml")
    assert policy_file.exists()
    with open(policy_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "thresholds" in data
    assert "rules" in data
    assert len(data["rules"]) >= 3
    assert data["rules"][0]["action"] == "REDIRECT_TO_DECOY"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables must override YAML configuration settings."""
    monkeypatch.setenv("GRAPHGUARD_ENV", "testing")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka-cluster:9092")

    cfg = load_config(config_path="config/default.yaml")
    assert cfg.system.env == "testing"
    assert cfg.database.name == "test_db"
    assert cfg.bus.bootstrap_servers == "kafka-cluster:9092"
