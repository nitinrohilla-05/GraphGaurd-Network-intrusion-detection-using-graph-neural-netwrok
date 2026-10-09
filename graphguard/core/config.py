"""Configuration models and loader for GraphGuard."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    env: str = "development"
    log_level: str = "INFO"
    seed: int = 42


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    threshold_tau: float = 0.70
    window_duration_seconds: int = 30
    slide_interval_seconds: int = 10
    feature_dim: int = 16
    open_set_energy_threshold: float = -5.0


class ContainmentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    k_hops: int = 2
    adam_steps: int = 200
    adam_lr: float = 0.05
    lambda1: float = 1.0
    lambda2: float = 0.5
    mask_threshold: float = 0.50
    max_escalation_rounds: int = 3
    verification_windows: int = 2


class RollbackConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    interval_seconds: int = 30
    consecutive_benign_windows: int = 3
    benign_confidence_threshold: float = 0.85
    max_evasion_score: float = 0.20
    watchlist_duration_hours: int = 24


class BusConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bootstrap_servers: str = "localhost:9092"
    internal_servers: str = "redpanda:29092"
    topics: Dict[str, str] = Field(
        default_factory=lambda: {
            "flows_raw": "flows.raw",
            "graph_windows": "graph.windows",
            "detections": "detections",
            "actions_desired": "actions.desired",
            "actions_applied": "actions.applied",
            "verifications": "verifications",
            "reactions": "reactions",
            "dlq": "dlq",
        }
    )


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    host: str = "localhost"
    port: int = 5432
    name: str = "graphguard"
    user: str = "graphguard"
    password: Optional[str] = None


class RedisConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class GraphGuardConfig(BaseModel):
    """Aggregated configuration model for GraphGuard runtime."""

    model_config = ConfigDict(extra="ignore")

    system: SystemConfig = Field(default_factory=SystemConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    containment: ContainmentConfig = Field(default_factory=ContainmentConfig)
    rollback: RollbackConfig = Field(default_factory=RollbackConfig)
    bus: BusConfig = Field(default_factory=BusConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    default_enforcer: str = "DryRunEnforcer"


def load_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
) -> GraphGuardConfig:
    """Load configuration from YAML and environment variables."""
    # Load .env file
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        # Default lookups for .env
        root_env = Path(".env")
        if root_env.exists():
            load_dotenv(root_env, override=False)

    data: Dict[str, Any] = {}

    target_path = Path(config_path) if config_path else Path("config/default.yaml")
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, dict):
                data = yaml_content

    # Environment variable overrides
    system_data = data.setdefault("system", {})
    if env_val := os.getenv("GRAPHGUARD_ENV"):
        system_data["env"] = env_val
    if log_val := os.getenv("LOG_LEVEL"):
        system_data["log_level"] = log_val

    bus_data = data.setdefault("bus", {})
    if k_boot := os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        bus_data["bootstrap_servers"] = k_boot
    if k_int := os.getenv("KAFKA_INTERNAL_SERVERS"):
        bus_data["internal_servers"] = k_int

    db_data = data.setdefault("database", {})
    if p_host := os.getenv("POSTGRES_HOST"):
        db_data["host"] = p_host
    if p_port := os.getenv("POSTGRES_PORT"):
        db_data["port"] = int(p_port)
    if p_db := os.getenv("POSTGRES_DB"):
        db_data["name"] = p_db
    if p_user := os.getenv("POSTGRES_USER"):
        db_data["user"] = p_user
    if p_pw := os.getenv("POSTGRES_PASSWORD"):
        db_data["password"] = p_pw

    redis_data = data.setdefault("redis", {})
    if r_host := os.getenv("REDIS_HOST"):
        redis_data["host"] = r_host
    if r_port := os.getenv("REDIS_PORT"):
        redis_data["port"] = int(r_port)

    if enf := os.getenv("DEFAULT_ENFORCER"):
        data["default_enforcer"] = enf

    return GraphGuardConfig.model_validate(data)
