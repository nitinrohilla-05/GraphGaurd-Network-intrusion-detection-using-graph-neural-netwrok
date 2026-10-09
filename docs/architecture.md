# GraphGuard — Architecture & System Design

## 1. Overview & Mission

GraphGuard is an autonomous network intrusion containment platform powered by Graph Neural Networks (GNNs). Unlike conventional Intrusion Detection Systems (IDS) that either generate alert fatigue or perform blunt host-level quarantine, GraphGuard solves for the **minimal disruption response** that contains an attack while preserving legitimate network traffic.

### Core Philosophy: The CVC Loop
GraphGuard's foundational contribution is the **Counterfactual → Verify → Contain (CVC)** loop:
1. **Counterfactual Optimization (N1)**: When an attack is flagged, the system identifies the minimal cut in the communication multi-graph by optimizing a continuous edge mask $M \in [0, 1]^E$ against the GNN loss and business asset criticality costs.
2. **Model-in-the-Loop Verification (N2)**: Actions are not assumed to succeed; subsequent traffic windows are re-scored by the detector to verify threat containment or trigger escalation.
3. **Graded Containment & Response Analysis (N3, N4, N5, N6)**:
   - **Graded Actions (N3)**: Selective drops, QoS rate limiting, or SDN diversion to decoy honeypots based on cost and confidence.
   - **What-If Rollback (N4)**: Continuous simulation to safely restore false alarms when traffic proves benign.
   - **Forecast-Aware Cut (N5)**: Link prediction to preemptively cover lateral movement targets with virtual edges.
   - **Intervention Response Analysis (N6)**: Detection of evasive adversarial reaction (e.g. port hopping, fallback targets).

---

## 2. Microservice Architecture

GraphGuard is designed as an event-driven, horizontally scalable microservice ecosystem communicating over Kafka/Redpanda:

```
[ Network Traffic / PCAP ]
            │
            ▼
┌───────────────────────┐
│     ingest service    │ ─── (flows.raw) ───┐
└───────────────────────┘                     │
                                              ▼
                                   ┌───────────────────────┐
                                   │  graph-builder service│
                                   └───────────────────────┘
                                              │ (graph.windows)
                                              ▼
                                   ┌───────────────────────┐
                                   │   inference service   │
                                   └───────────────────────┘
                                              │ (detections)
                                              ▼
                                   ┌───────────────────────┐
                                   │  containment service  │ (N1-N6 orchestrator)
                                   └───────────────────────┘
                                              │ (actions.desired)
                                              ▼
                                   ┌───────────────────────┐
                                   │   enforcer service    │ ─── (actions.applied)
                                   └───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          [ OsKen OpenFlow Controller ]                     [ Linux nftables / DryRun ]
```

### Microservice Directory & Responsibilities
| Service | Role | Scaling Strategy |
|---|---|---|
| `ingest` | Flow parsing (NetFlow v9/IPFIX, CICFlowMeter, PCAP replay) | 1 per capture interface / span port |
| `graph-builder` | Sliding time windows (30–60s), feature normalization, shard aggregation | Sharded by subnet (`shard_id`) |
| `inference` | Batched GNN forward passes, open-set uncertainty scoring | Horizontal replica scaling via consumer lag |
| `containment` | Orchestrates N1 cut solver, N2 verifier, N3 action policy, N4-N6 | Distributed leader election via Redis locks |
| `enforcer` | Idempotent enforcement and periodic state reconciliation (10s loop) | 1 per network domain / controller |
| `api` | FastAPI REST endpoints & WebSocket telemetry feeds | Horizontal behind load balancer |
| `dashboard` | React + Cytoscape.js live graph management console | Static CDN distribution |

---

## 3. Data Topologies & Storage Tiers

### Streaming Bus (Redpanda / Kafka)
- **`flows.raw`**: Individual flow records parsed from network sensors.
- **`graph.windows`**: Pure-Python serialized graph snapshots partitioned by `shard_id`.
- **`detections`**: Inferences with node/edge maliciousness scores and epistemic uncertainty.
- **`actions.desired`**: Actions published by containment orchestrator.
- **`actions.applied`**: Acknowledgements and active rule IDs published by enforcers.
- **`verifications`**: Containment verification verdicts and score history.
- **`reactions`**: Post-intervention evasion assessments.
- **`dlq`**: Dead-letter queue for unparseable or poisoned messages.

### Persistence (TimescaleDB / PostgreSQL 16)
- Time-series flow metrics and edge telemetry.
- Incident lifecycle records and post-incident verification histories.
- Cryptographically auditable event log (`AuditEvent`).

### State & Cache (Redis 7)
- Distributed locks for incident orchestration (prevents duplicate cut calculations).
- Watchlist registry with TTL for rolled-back sources (N4).
- Active enforcer rule caches for reconciliation.

---

## 4. Edge Identity & Graph Formulation

Network flows inherently form a **directed multi-graph**: two hosts can simultaneously maintain connections across different services (e.g., DNS port 53, HTTPS port 443, SSH port 22).

GraphGuard identifies every edge uniquely via `EdgeKey`:
$$\text{EdgeKey} = (\text{src\_ip}, \text{dst\_ip}, \text{dst\_port}, \text{proto})$$
with canonical string identifier:
$$\text{id} = \text{src\_ip} > \text{dst\_ip} : \text{dst\_port} / \text{proto}$$

All models, cut sets, actions, and cost computations refer directly to `EdgeKey`, ensuring deterministic indexing and native JSON serializability across message buses and persistence stores.
