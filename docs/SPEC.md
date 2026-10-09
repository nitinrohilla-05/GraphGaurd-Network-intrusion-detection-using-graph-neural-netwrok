# GraphGuard — Master Specification (for Antigravity agents)

> Save this file as `docs/SPEC.md` in the repository. Every agent task must read it first.
> Owner/inventor: the student (me). Agents implement; they do not change the novel features without approval.

---

## 1. Mission

Build **GraphGuard**, a scalable, production-style Graph Neural Network (GNN) system that **detects** network attacks and **contains** them with the smallest possible disruption to legitimate traffic.

- Hosts (IP endpoints) = **nodes**. Network flows = **edges** with features (bytes, packets, duration, src/dst port, protocol, TCP flags, timestamps).
- GNN detection alone (E-GraphSAGE, Anomal-E) already exists and is **only the baseline**.
- GraphGuard's original contribution is the **CVC loop — Counterfactual → Verify → Contain**: the GNN detector itself computes the minimum response, tests it, and decides whether to keep, strengthen or undo it.

## 2. Non-goals (do NOT build these as headline features)

- Generic "predict next target and block it" without the cost-aware graph cut (N1)
- Simple time-based block → retest → unblock
- Rollback triggered only because "the anomaly stopped"
- LLM-generated IDS rules, federated learning, digital-twin change testing

---

## 3. Original features (core of the project — implement exactly)

### N1. Cost-aware counterfactual minimal cut — `graphguard/containment/cut_solver.py`
- Input: current `GraphWindow` + flagged edges/nodes from the detector.
- Extract the k-hop subgraph (k = GNN depth) around flagged targets.
- Learn a soft mask `M = sigmoid(θ)` over candidate edges; optimise with Adam (≤ 200 steps):

  ```
  L = L_flip + λ1 · Σ_e cost(e) · (1 − M_e) + λ2 · Σ_e (1 − M_e)
  ```
  - `L_flip`: BCE pushing the detector's malicious score on flagged targets below threshold τ when the graph is re-scored with edge weights M.
  - `cost(e) = α · criticality(src, dst) + β · normalised_benign_volume(e)`; criticality from `config/assets.yaml` (gateway, DNS, DHCP, DB, domain controller = high).
- Output: `CutSet` = edges with `M_e < 0.5`, ranked by importance, each with its cost.
- Also implement `GreedyCutSolver` (remove edges by gradient score until the prediction flips) as a baseline. Both implement the `CutSolver` interface.

### N2. Model-in-the-loop verification — `graphguard/containment/verifier.py`
- After enforcement, rebuild the graph from the next traffic window and re-score it with the **same** detector.
- Verdict `CONTAINED` if all flagged targets score < τ for V consecutive windows.
- Otherwise rerun N1 on the new graph (max R rounds), then `ESCALATED` (isolate host, flag for admin review).
- Persist a `VerificationRecord`: per-round scores before/after, edges cut, rules installed, timings.

### N3. Graded per-edge actions — `graphguard/containment/action_policy.py`
- For each edge in a `CutSet` choose `DROP`, `RATE_LIMIT` or `REDIRECT_TO_DECOY` from (detector confidence, edge cost, target criticality):
  - high confidence + low cost → `DROP`
  - uncertain → `RATE_LIMIT`
  - edge toward a high-criticality target → `REDIRECT_TO_DECOY` (SDN rewrites destination to the honeypot host)
- Start as a configurable rule table (`config/action_policy.yaml`); keep the interface so it can become a learned policy later.

### N4. What-if rollback — `graphguard/containment/rollback.py`
- Every T seconds, for each active action, build a **what-if graph**: latest window + the blocked edges re-inserted using the source's most recent observed flow features.
- Re-score with the detector. If benign with confidence ≥ c for W consecutive checks **and** N6 evasion score is low → revert the action, log it, add the source to a **watchlist** (lower re-block threshold for X hours).
- Track rollback errors (source re-flagged after rollback).

### N5. Forecast-aware cut — `graphguard/forecast/link_predictor.py`
- Temporal link-prediction head (shares the detector's encoder) scores candidate future edges from a flagged source to hosts it has not yet contacted.
- Top-k predicted edges above probability p are added to N1's candidate set as **virtual edges** with their own cost.
- Virtual-edge actions may only be `RATE_LIMIT` or `REDIRECT_TO_DECOY` — never a hard `DROP`.

### N6. Intervention Response Analysis — `graphguard/containment/reaction_analyzer.py`
- Treat every applied action as an experiment. Compare each affected source's behaviour before vs. after the action.
- Reaction features: retries to the blocked dst:port; new destinations contacted (especially those in N5's predicted set); port/protocol switching; fan-out change; timing regularity; share of traffic hitting the decoy.
- Output `evasion_score ∈ [0, 1]` from a small classifier/GNN on the post-action subgraph.
  - High → attacker is routing around the block → confirm, strengthen (re-run N1 including new edges), escalate for review.
  - Low + benign re-score → feeds N4 rollback.
- Training data from the testbed: (a) block benign apps and record their reactions (backoff, quiet failure); (b) block attack scripts written with fallback targets and record their evasion.

---

## 4. Supporting features (standard, needed for a complete product)
- Detection: E-GraphSAGE edge classifier (baseline) + temporal sliding-window variant; open-set `UNKNOWN` class via uncertainty (energy score or MC-dropout).
- Incident card: which edges were cut and why, what traffic was preserved; optional LLM plain-English summary + MITRE ATT&CK tactic guess, labelled "AI-generated".
- Dashboard: live graph (attacked nodes red, cut edges dashed, decoy redirects highlighted), incident timeline, active actions with TTL/rollback status, metrics, **auto / approve** mode toggle.
- API security: JWT auth, roles (viewer / operator / admin), audit log of every action and approval.

---

## 5. Data model — `graphguard/core/models.py` (pydantic v2)
`FlowRecord`, `GraphWindow` (wraps PyG `Data`: `edge_index`, `edge_attr`, `node_ids`, `t_start`, `t_end`, `shard_id`), `Detection`, `Incident`, `CutSet`, `Action` (`type`, `match{src,dst,dport,proto}`, `created_at`, `status`, `virtual: bool`), `VerificationRecord`, `ReactionReport`, `AuditEvent`.

## 6. Interfaces — `graphguard/core/interfaces.py` (ABCs)
```
Detector.predict(window) -> Detection(edge_scores, node_scores, uncertainty)
CutSolver.solve(window, detection, cost_fn, extra_candidates=None) -> CutSet
ActionPolicy.decide(cut_set, detection) -> list[Action]
Enforcer.apply(actions) -> list[str]        # idempotent, returns rule ids
Enforcer.revert(action_id)
Enforcer.reconcile(desired: list[Action])   # makes installed rules match desired state
Verifier.step(incident, window) -> Verdict
ReactionAnalyzer.analyze(source, pre_windows, post_windows) -> ReactionReport
RollbackManager.tick(now)
LinkPredictor.predict(window, source, k) -> list[(dst, dport, prob)]
```
Enforcer implementations: `OsKenEnforcer` (OpenFlow 1.3), `NftablesEnforcer`, `DryRunEnforcer` (default).

---

## 7. Scalability architecture

Microservices, stateless where possible, all communicating over a message bus. Docker Compose for development; Kubernetes manifests for scale.

| Service | Role | Scaling |
|---|---|---|
| `ingest` | pcap replay / live capture → flow records (CICFlowMeter or nProbe NetFlow v9/IPFIX) | 1 per capture point |
| `graph-builder` | sliding windows (30–60 s), feature normalisation, incremental updates | sharded by subnet (`shard_id`); cross-shard edges stitched by a merger |
| `inference` | batched GNN scoring; TorchScript/ONNX export optional; GPU optional | horizontal; HPA on Kafka consumer lag |
| `containment` | orchestrator for N1–N6 | one leader per incident via Redis lock (no double actions) |
| `enforcer` | applies/reverts rules; reconciliation loop every 10 s | 1 per controller/firewall |
| `api` | FastAPI REST + WebSocket | horizontal behind load balancer |
| `dashboard` | React + Cytoscape.js | static, CDN-able |

- **Bus:** Kafka (or Redpanda). Topics: `flows.raw`, `graph.windows`, `detections`, `actions.desired`, `actions.applied`, `verifications`, `reactions`, `dlq` (dead-letter). Partition key = `shard_id`.
- **Storage:** PostgreSQL + TimescaleDB (flows, incidents, metrics, audit), Redis (live state, locks, watchlist).
- **Reliability:** idempotent consumers, at-least-once processing, back-pressure (pause consumers when lag > threshold), action TTL + reconciliation so a crashed service never leaves stale rules.
- **Observability:** Prometheus metrics (`gg_flows_ingested_total`, `gg_detection_latency_seconds`, `gg_time_to_contain_seconds`, `gg_active_actions`, `gg_collateral_damage_ratio`), Grafana dashboards, OpenTelemetry traces across services, structured JSON logs.
- **Load testing:** `tcpreplay` at 1k / 10k / 50k flows/sec; report throughput, p50/p95/p99 latency, consumer lag, CPU/RAM.
- **Kubernetes:** `deploy/k8s/` with kustomize overlays (`dev`, `scale`); HPA for `inference` and `api`.

---

## 8. Repository layout
```
graphguard/
  core/        models.py, interfaces.py, config.py
  ingest/      pcap_replay.py, flow_exporter.py, kafka_io.py
  graph/       window_builder.py, features.py, shard_merger.py
  detect/      egraphsage.py, temporal.py, open_set.py, train.py, registry.py
  forecast/    link_predictor.py
  containment/ cut_solver.py, action_policy.py, verifier.py, rollback.py, reaction_analyzer.py, orchestrator.py
  enforce/     osken_app.py, osken_enforcer.py, nftables_enforcer.py, dryrun.py, reconciler.py
  api/         main.py, auth.py, routes/, ws.py
  eval/        metrics.py, baselines.py, ablations.py, plots.py, loadtest.py
dashboard/     React app
testbed/       mininet_topo.py, scenarios/*.py, benign_traffic.py
config/        default.yaml, assets.yaml, action_policy.yaml
deploy/        docker-compose.yml, k8s/
tests/         unit/, integration/
docs/          SPEC.md, architecture.md, decisions.md, runbook.md
Makefile, pyproject.toml, README.md
```

## 9. Stack
Python 3.11, PyTorch + PyTorch Geometric, FastAPI, pydantic v2, Kafka/Redpanda, PostgreSQL/TimescaleDB, Redis, **os-ken** SDN controller (maintained Ryu fork, runs in its **own container** with its own Python version), Mininet, React + Cytoscape.js, Prometheus, Grafana, OpenTelemetry, Docker Compose, Kubernetes (kustomize).

## 10. Environment
- Linux host required for Mininet/Open vSwitch: Ubuntu 22.04/24.04, or WSL2 / VM on Windows.
- Everything except Mininet runs in Docker.

## 11. Data and testbed
- **Offline:** University of Queensland NetFlow-v2 datasets — NF-UNSW-NB15-v2, NF-ToN-IoT-v2, NF-BoT-IoT-v2, NF-CSE-CIC-IDS2018-v2. Time-ordered splits, no leakage, class-imbalance handling.
- **Live lab (isolated, never connected to real networks):** Mininet + Open vSwitch + os-ken, ≥ 10 hosts incl. gateway, DNS, web, DB, file server and one decoy/honeypot host. Continuous benign traffic (HTTP, DNS, file copy) at all times.
- **Attack scenarios (lab only):**
  - port scan (nmap)
  - SYN flood (hping3)
  - SSH brute force on a test account
  - lateral-movement chain host → host → DB
  - an *evasive* script that switches to fallback targets when blocked (needed for N6)

## 12. Metrics — `graphguard/eval/metrics.py` (computed automatically every run)
- **Detection:** precision, recall, F1, ROC-AUC per class.
- **Collateral Damage Rate** = benign flows blocked / all benign flows of affected hosts.
- **Containment:** containment success rate; time to contain; edges cut and rules installed per incident.
- **N5 prediction usefulness:** % of the attacker's actual next edges pre-covered.
- **N6:** evasion-detection precision/recall; false-alarm restore rate.
- **N4:** rollback precision; rollback error rate.
- **System:** flows/sec, latency p50/p95/p99, consumer lag.
- **Baselines:** no response | isolate whole host | block every flagged flow | greedy cut.
- **Ablations:** full system minus N2, N3, N4, N5, N6 (one at a time).

## 13. Engineering rules
- Type hints everywhere; ruff + mypy; pytest unit tests for every module (mock the detector in containment tests); integration tests per phase.
- Every phase runs end-to-end with one command (`make pN`).
- `DryRunEnforcer` by default; real enforcement only inside the Mininet lab.
- Fixed random seeds; configs in YAML; no hard-coded paths or secrets (`.env` + `.env.example`).
- **Repository stays PRIVATE** (patent may be filed on N1–N6). Never add a public remote, never publish packages, never paste code into public sites.
- `docs/decisions.md`: dated entry for every design decision, stating whether the idea came from the owner or the agent (inventorship record).

---

## 14. Phases (one Antigravity task per phase)

Each phase: **(1)** produce an implementation-plan artifact (files, interfaces, tests, commands) and wait for approval; **(2)** implement; **(3)** run tests and the "Done when" check in the terminal; **(4)** produce a walkthrough artifact with results, screenshots/logs and known issues; **(5)** append to `docs/decisions.md`.

| Phase | Work | Done when |
|---|---|---|
| P0 | Architecture doc, repo skeleton, interfaces, configs, Makefile, compose with Kafka/Postgres/Redis | `make test` passes on stubs; `docker compose up` healthy |
| P1 | E-GraphSAGE detector on one NF-v2 dataset + model registry | per-class F1 report saved; model versioned |
| P2 | N1 cut solver offline + greedy + baselines | collateral-damage comparison table generated |
| P3 | Mininet/os-ken lab + enforcers + reconciler + N2 live | scan scenario contained and verified in < 60 s |
| P4 | N3 graded actions incl. decoy redirect | attacker reaches honeypot; benign traffic untouched |
| P5 | N5 link predictor + virtual edges in N1 | prediction-usefulness metric reported on lateral-movement scenario |
| P6 | N6 reaction analyzer trained on lab reactions | evasive script confirmed; blocked benign app cleared |
| P7 | N4 what-if rollback + watchlist | false alarm auto-restored; paused attacker re-blocked |
| P8 | Kafka streaming, sharded graph builder, HPA-ready services, load test | throughput + p95 latency reported at 1k/10k flows/sec |
| P9 | API (auth, roles, audit) + dashboard | browser walkthrough shows live incident end-to-end |
| P10 | Full evaluation, ablations, plots, k8s manifests, runbook, report chapter draft | all metrics + plots in `eval/results/` |

**Parallel agents:** allowed only from P8 onwards, and only on non-overlapping folders (e.g. one agent on `dashboard/`, one on `eval/`, one on `deploy/k8s/`). Before P8, one agent at a time.
