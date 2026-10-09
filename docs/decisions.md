# GraphGuard — Decisions & Inventorship Log

All architectural choices, feature specifications, and implementation decisions are recorded here with timestamps and attribution (Owner vs Agent) per workspace rule 10.

---

## 2026-10-08 — Phase P0 Architecture & Core Contracts

### Decision 1: Edge Identity & Discrete Ports
- **Attribution:** Owner
- **Decision:** Introduce a dedicated `EdgeKey` model (`src_ip`, `dst_ip`, `dst_port`, `proto`) with a deterministic, canonical string identifier format: `"{src_ip}>{dst_ip}:{dst_port}/{proto.lower()}"`. Same host pair communicating across different ports or protocols are strictly treated as distinct edges.
- **Rationale:** Network flows are multi-graphs where connections between identical IPs serve different services (e.g., DNS port 53 vs HTTP port 80). Using string IDs ensures seamless JSON serialization and clean Redis/Kafka message passing without complex tuple key serialization workarounds.

### Decision 2: CutSet Representation & Cost Function Signature
- **Attribution:** Owner
- **Decision:** `CutSet` holds a list of `CutEdge` objects (`edge: EdgeKey`, `mask_weight: float`, `cost: float`, `virtual: bool`). The cost function signature is standardized as `Callable[[EdgeKey, GraphWindow], float]`.
- **Rationale:** Avoid tuple-keyed dictionaries in data models for zero-loss JSON serialization across API and storage boundaries.

### Decision 3: Action Lifecycle, Expiration (TTL), and Incident Status
- **Attribution:** Owner
- **Decision:** Action models must include `incident_id` and `expires_at: Optional[datetime]` (TTL). `ActionStatus` is modeled as an Enum (`PENDING`, `APPLIED`, `REVERTED`, `EXPIRED`, `FAILED`). `Incident.status` is modeled as an Enum (`OPEN`, `CONTAINED`, `ESCALATED`, `CLOSED`). `DryRunEnforcer.reconcile` automatically detects and purges expired actions.
- **Rationale:** Prevents orphaned firewall rules or permanent blocks if a controller or microservice crashes, satisfying zero-stale-rule resilience requirements.

### Decision 4: Pure-Python GraphWindow with Lazy PyG Conversion
- **Attribution:** Owner
- **Decision:** `GraphWindow` fields remain pure-Python (lists, primitives, datetimes) to ensure universal serialization and avoid Torch C++ dependency overhead in non-inference services. PyG `Data` instances are generated on-demand via a `.to_pyg()` method.
- **Rationale:** Keeps ingestion, bus dispatch, and API services lightweight while providing seamless integration for the GNN detector in inference stages.

### Decision 5: Dual-Listener Kafka Architecture via Redpanda
- **Attribution:** Owner
- **Decision:** Redpanda is configured with two distinct listeners: `internal://redpanda:29092` for Docker network inter-service traffic and `external://localhost:9092` for host machine tools and integration tests.
- **Rationale:** Eliminates host-port binding conflicts and network resolution issues between containerized services and host test runners.

### Decision 6: TimescaleDB PostgreSQL Configuration & Secret Isolation
- **Attribution:** Owner
- **Decision:** PostgreSQL service uses image `timescale/timescaledb:latest-pg16`. Credentials (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`) are sourced from `.env` via Docker Compose variable interpolation. `.env` is added to `.gitignore` and only `.env.example` is committed to version control.
- **Rationale:** Adheres to workspace rule 9 (no secrets in repository).

### Decision 7: Static Typing & Strict Verification Pipeline
- **Attribution:** Owner
- **Decision:** Enforce `mypy` in strict mode on `graphguard/core` with a dedicated `make typecheck` target. `make p0` is defined as `lint` + `typecheck` + `test`.
- **Rationale:** Guarantees airtight type safety across core data structures and interfaces before downstream feature modules depend on them.

### Decision 8: Repository Skeleton Placeholders
- **Attribution:** Owner
- **Decision:** Pre-create all folders specified in SPEC Section 8 (`testbed/`, `dashboard/`, `deploy/k8s/`, `graphguard/eval/`) with informative placeholder `README.md` files indicating which project phases will populate them.
- **Rationale:** Prevents structural drift as phases advance.

### Decision 9: Docker Compose Environment Variable Binding & Strict Password Guard
- **Attribution:** Owner
- **Decision:** Makefile `up`, `down`, and `ps` targets must explicitly pass `--env-file .env` when executing `docker compose -f deploy/docker-compose.yml`. In `deploy/docker-compose.yml`, Postgres password uses strict parameter expansion `${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}` with no insecure fallback.
- **Rationale:** The compose file lives in `deploy/`, so Docker Compose would otherwise fail to locate `.env` at repository root. Enforcing explicit error on missing `POSTGRES_PASSWORD` prevents silent startup with default or missing credentials.

### Decision 10: Unified EdgeKey Usage across Models & ActionMatch Deprecation
- **Attribution:** Owner
- **Decision:** Replace `ActionMatch` directly with `EdgeKey` (`Action.match: EdgeKey`).
- **Rationale:** Eliminates redundant data models and ensures a single canonical definition of edge identity throughout GraphGuard.

### Decision 11: EdgeKey Validation with ipaddress and Lowercase Protocol
- **Attribution:** Owner
- **Decision:** `EdgeKey` includes a Pydantic field validator that lowercases `proto` and validates `src_ip` and `dst_ip` using Python's standard `ipaddress.ip_address` module.
- **Rationale:** Guarantees strict IP address hygiene and prevents malformed protocol strings from compromising graph construction and flow hashing.

### Decision 12: GraphWindow Edge Consistency & Constant Node Features in to_pyg()
- **Attribution:** Owner
- **Decision:** `GraphWindow` enforces a model validator verifying `len(edge_index) == len(edge_attr) == len(edges)` with matching ordering. `to_pyg()` initializes node features with constant ones (`torch.ones((num_nodes, 1))`) and documents this explicitly in its docstring.
- **Rationale:** Ensures mathematical and structural consistency across PyG conversion when raw node attributes are not yet assigned.

### Decision 13: Core Interface Inventory (8 ABCs)
- **Attribution:** Owner
- **Decision:** Standardize SPEC Section 6 interfaces to exactly 8 Abstract Base Classes: `Detector`, `CutSolver`, `ActionPolicy`, `Enforcer`, `Verifier`, `ReactionAnalyzer`, `RollbackManager`, `LinkPredictor`. Tests must explicitly verify all 8 ABCs.
- **Rationale:** Precise contract coverage avoiding miscount or ambiguity between ABCs and concrete implementations.

---

## 2026-10-08 — Phase P1 Detector, Features & Model Registry

### Decision 14: Real Dataset Sourcing & Stratified Test Slice
- **Attribution:** Owner
- **Decision:** Never generate synthetic data for training. The real NetFlow dataset is placed at `data/raw/<file>.csv` (ignored by git via `.gitignore`). The file path is configured in `config/detection.yaml`. A small stratified slice (`data/sample_nf_v2.csv`) is preserved solely for unit testing.
- **Rationale:** GNN intrusion detection requires authentic flow distributions, realistic class imbalances, and genuine graph topologies.

### Decision 15: Flow Timestamp Inspection & Chunk-Based Fallback
- **Attribution:** Owner
- **Decision:** Inspect the CSV header first to detect flow timestamp columns (e.g. `FLOW_START_MILLISECONDS`). If present, build temporal sliding windows; if absent, construct windows from fixed-size consecutive row chunks (e.g. 50k flows) preserving original row order for time-ordered splits. Never fabricate timestamps.
- **Rationale:** Protects temporal integrity and prevents synthetic time leakage.

### Decision 16: NetFlow Numeric Features & Leakage Exclusion
- **Attribution:** Owner
- **Decision:** Extract numeric NetFlow features directly from the dataset with log-scaling and standard scaling (fitted strictly on the training split). Exclude identifier columns (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `Label`, `Attack`) from edge features.
- **Rationale:** Prevents models from memorizing specific attacker IP addresses or ephemeral source ports, ensuring generalizable topological learning.

### Decision 17: IP-Shuffle Leakage Verification
- **Attribution:** Owner
- **Decision:** Evaluate test performance with and without randomly remapped IP addresses (IP-shuffle) to verify that GNN graph learning relies on structural communication patterns rather than static IP memorization.
- **Rationale:** Rigorous evaluation against IP overfitting.

### Decision 18: Tabular Baseline Benchmark
- **Attribution:** Owner
- **Decision:** Train a tabular baseline (RandomForest / XGBoost) on the exact same edge feature vectors to directly benchmark GNN graph benefits against flat flow classifiers.
- **Rationale:** Provides an honest baseline comparison to demonstrate graph structural value.

### Decision 19: Validation-Calibrated Decision & Energy Thresholds
- **Attribution:** Owner
- **Decision:** Calibrate detection threshold $\tau$ and the open-set energy threshold dynamically on the validation split rather than hardcoding static values.
- **Rationale:** Adapts to specific dataset distributions and maximizes F1 while controlling false positives.

### Decision 20: Leave-One-Attack-Out Open-Set Evaluation
- **Attribution:** Owner
- **Decision:** Perform a leave-one-attack-out experiment by withholding one attack class from training, evaluating open-set AUROC for identifying novel attacks as `UNKNOWN` via energy scores.
- **Rationale:** Validates the detector's capability to identify novel zero-day threats.

### Decision 21: Granular Metric Reporting
- **Attribution:** Owner
- **Decision:** Report sample and class distributions across train/val/test splits, together with macro-F1, weighted-F1, and per-class F1 tables.
- **Rationale:** Essential for analyzing severe network traffic class imbalances.

### Decision 22: MC-Dropout Optional & Shard Merger Deferral
- **Attribution:** Owner
- **Decision:** Make Monte Carlo Dropout optional (disabled by default for fast inference); keep `shard_merger.py` as a placeholder stub until Phase P8 streaming integration.
- **Rationale:** Keeps Phase P1 focused on the detector core without unnecessary runtime overhead.

### Decision 23: Fixed Random Seeds & Training Curve Artifacts
- **Attribution:** Owner
- **Decision:** Enforce deterministic random seeds across PyTorch, NumPy, and Scikit-Learn; log training/validation loss curves directly into the model registry directory alongside model weights.
- **Rationale:** Complete reproducibility and auditability.

### Decision 24: Per-Flow Source Randomization (Hub-Dependence Check)
- **Attribution:** Owner
- **Decision:** Replace uniform IP permutation with per-flow source randomization: for each test flow independently, replace `IPV4_SRC_ADDR` with a random address drawn from a large pool (e.g. `10.128.0.0/9`), rebuild windows, re-run inference, and report F1 with vs without as a hub-dependence metric.
- **Rationale:** Because node features are constant ones, uniform bijective IP permutation leaves the graph structurally isomorphic. Per-flow randomization breaks hub structure and directly tests whether the model depends on high-degree source hubs.

### Decision 25: Dataset-Driven Attack Categories & Reconnaissance Holdout
- **Attribution:** Owner
- **Decision:** Read attack category names dynamically from the dataset's `Attack` (or attack label) column, print actual category counts, and select "Reconnaissance" (or closest existing category) as the held-out class for open-set evaluation. Never hardcode fictitious class names.
- **Rationale:** Guarantees open-set evaluation reflects genuine ground-truth taxonomy.

### Decision 26: Binary Attack Score Formulation & Threshold Calibration
- **Attribution:** Owner
- **Decision:** Formulate binary attack score as $1 - P(\text{benign})$. Calibrate threshold $\tau$ on the validation split by maximizing F1 on this binary attack score, while still calculating and reporting multi-class per-class F1.
- **Rationale:** Maximizes threat containment recall while rigorously tracking fine-grained class performance.

### Decision 27: Stratified Subsampling for Tabular Baseline
- **Attribution:** Owner
- **Decision:** If the full training split is too large for fast RandomForest training, perform stratified subsampling with a fixed seed and document the exact sample size in the walkthrough.
- **Rationale:** Ensures fast, reproducible training without distorting class priors.

### Decision 28: Row Order Chronological Assumption on Datasets Lacking Timestamps
- **Attribution:** Owner
- **Decision:** When the dataset lacks explicit flow timestamp columns, state explicitly in the walkthrough and report that time-ordered chunk splits assume CSV row ordering is chronological, which cannot be independently verified.
- **Rationale:** Intellectual honesty and methodological transparency in absence of explicit flow start timestamps.

---

## 2026-10-09 — Phase P2 Cost-Aware Counterfactual Cut Solver (N1)

### Decision 29: Candidate Edge Restriction via tau_low and Incident Source Filtering
- **Attribution:** Owner
- **Decision:** N1 candidate cut edges are strictly restricted to: (a) edges with detector attack score >= tau_low (default 0.30), OR (b) edges incident to a source node of a flagged edge. Edges between two unflagged nodes with score < tau_low are never candidates for cutting.
- **Rationale:** Prevents the optimizer from evaluating or modifying unrelated benign background communication.

### Decision 30: Hard-Removal Verification & Iterative Remediation
- **Attribution:** Owner
- **Decision:** After thresholding soft mask M_e < 0.5, candidate cut edges are hard-removed from the graph. The remaining subgraph is re-scored with the detector, requiring every remaining unblocked edge in the region to score < tau. If this condition fails, iteratively add the highest-scoring remaining edge to the cut and re-check (up to 10 iterations).
- **Rationale:** Eliminates soft-mask approximation gaps and guarantees concrete dataplane containment.

### Decision 31: Mask-Weighted Message Passing Equivalence
- **Attribution:** Owner
- **Decision:** Implement mask-weighted mean aggregation in the GNN message passing: sum(M_e * m_e) / (sum(M_e) + eps). Unit-test that setting M_e = 0 yields mathematically identical embeddings and scores to hard edge deletion.
- **Rationale:** Guarantees that continuous counterfactual optimization smoothly mirrors discrete physical graph cut decisions.

### Decision 32: Flow-Level Ground-Truth Evaluation Metrics
- **Attribution:** Owner
- **Decision:** Evaluate containment outcomes at the granular flow level (blocking an EdgeKey blocks all underlying flows): Attack Block Rate = true attack flows blocked / all true attack flows in region; Collateral Damage Rate = true benign flows blocked / all benign flows of affected hosts; Residual Attack Rate = true attack flows unblocked; Model-based containment check reported separately and clearly labeled; Solver latency (mean and p95) reported per incident.
- **Rationale:** Aligns network performance evaluation with true operational telemetry and collateral business impact.

### Decision 33: Data-Driven Asset Role Inference
- **Attribution:** Owner
- **Decision:** Infer asset roles dynamically from network traffic (e.g. hosts receiving most traffic on port 53 -> DNS, 3306/5432/1433 -> DB, top-degree internal host -> Gateway), saving to config/assets_inferred.yaml. Never invent arbitrary node roles.
- **Rationale:** Grounded real-world asset criticality modeling without manual inventory configuration.

### Decision 34: Source-Node Isolation for IsolateHost Baseline
- **Attribution:** Owner
- **Decision:** The IsolateHost containment baseline strictly isolates the source node of flagged malicious edges, cutting all its incident edges.
- **Rationale:** Reflects realistic network incident response where an infected machine is disconnected from the switch port.

### Decision 35: Lambda Hyperparameter Tradeoff Sweep & Frontier Plot
- **Attribution:** Owner
- **Decision:** Execute a grid sweep across at least 5 configurations of (lambda1, lambda2) to plot the Attack Block Rate vs Collateral Damage Rate frontier for N1, placing baselines as discrete comparison points.
- **Rationale:** Demonstrates the Pareto efficiency of cost-aware counterfactual optimization over heuristic cuts.

### Decision 36: Controlled Unit Tests with Mock Detector & Adversarial Cut Invariance
- **Attribution:** Owner
- **Decision:** Unit-test the solver on a hand-crafted graph with known ground truth and mock detector, including a test verifying that N1 does not cut benign neighbor edges merely to manipulate an attack edge score.
- **Rationale:** Verifies optimizer convergence and rules out adversarial gaming of graph structure.

### Decision 37: Data Provenance & Real Dataset Verification
- **Attribution:** Owner
- **Decision:** Explicitly state the exact provenance of all evaluation data: `data/raw/` is currently empty (0 files). Training and evaluation ran on `data/sample_nf_v2.csv` (300 flows, 14 columns, 2 consecutive 150-flow windows, 2 evaluated incidents, IP range `192.168.1.1` to `192.168.1.200` on `192.168.1.0/24`). No synthetic data is fabricated. When the full NF-UNSW-NB15-v2 CSV is placed in `data/raw/`, `config/detection.yaml` immediately loads it for training and evaluation.
- **Rationale:** Complete scientific honesty and provenance transparency; sample slice is strictly acknowledged and documented.

### Decision 38: P1 Detection Benchmark & Model Verification
- **Attribution:** Owner
- **Decision:** Fix multi-class label mapping in `train.py` so attack categories are preserved rather than collapsed to binary label 1. Report full detector metrics: GNN vs. RandomForest comparison, per-class F1 breakdown across all categories, per-flow source randomization hub-dependence test ($\Delta = +0.1000$), and open-set AUROC on held-out 'Reconnaissance' ($0.8110$).
- **Rationale:** Verifies detector generalization, structural dependency, and open-set anomaly detection capability prior to containment benchmarking.

### Decision 39: Model Containment Debugging & Post-Remediation Verification
- **Attribution:** Owner
- **Decision:** Diagnose and resolve the 0% Model Containment Success rate by ensuring post-remediation verification evaluates against the detector's calibrated threshold $\tau = 0.5227$ rather than an arbitrary 0.50 threshold, and enhancing the counterfactual objective with direct unblocked threat loss. Per-incident logs record: candidate count, cut size, remediation iterations used, maximum remaining unblocked edge score, and failure reason, confirming evaluation occurs strictly on the post-cut remaining graph.
- **Rationale:** Guarantees transparent post-action remediation accounting and exposes topological edge effects (e.g., GNN message passing score redistribution after edge deletion).

### Decision 40: Multi-Strategy Continuous Frontier Curves & Matched-ABR Collateral Damage
- **Attribution:** Owner
- **Decision:** Perform continuous hyperparameter sweeps across all three active containment strategies: $\tau \in [0.15, 0.95]$ for `BlockAllFlagged`, $\lambda_{\text{cost}} \in [0.0, 5.0]$ for `GreedyCut`, and $(\lambda_1, \lambda_2)$ for `N1 CounterfactualCut`. Plot continuous curves with Attack Block Rate on the x-axis and Collateral Damage Rate on the y-axis. Compute and report CDR at matched ABR levels (80%, 90%, 95%).
- **Rationale:** Ensures a mathematically fair comparison across the entire operating spectrum rather than arbitrary single-point baseline selections.

### Decision 41: Headline vs Secondary Table Schema Separation
- **Attribution:** Owner
- **Decision:** Remove "Total cost" completely from the headline containment comparison table because Total Cost is the objective quantity that N1 directly optimizes (making its inclusion in the primary comparison circular). Retain intervention size and cost metrics in a dedicated secondary table.
- **Rationale:** Enforces methodological neutrality and eliminates optimization circularity in headline evaluation.

### Decision 42: Statistical Rigor & Percentile Bootstrap Confidence Intervals
- **Attribution:** Owner
- **Decision:** Report exact incident counts ($N=2$) and 95% bootstrap confidence intervals (1000 resamples over incidents) for both Attack Block Rate and Collateral Damage Rate.
- **Rationale:** Prevents misleading precision claims on small sample slices and provides rigorous statistical bounds.

### Decision 43: Neutral Takeaways & Disadvantage Disclosure
- **Attribution:** Owner
- **Decision:** Eliminate all promotional or marketing language from benchmark takeaways. State results neutrally and explicitly document where N1 loses: N1 incurs orders-of-magnitude higher runtime latency (~250ms vs ~5ms for heuristic baselines) due to iterative gradient descent, and can suffer from conservative under-blocking when high regularizer weights suppress necessary edge cuts.
- **Rationale:** Adheres to rigorous engineering and scientific standards of objective evaluation.

### Decision 44: Success-Condition Refinement & Post-Remediation Score Drift Isolation
- **Attribution:** Owner
- **Decision:** Separate post-cut verification into two distinct metrics:
  (a) True Containment: Every remaining edge that was a candidate (initial score $\ge \tau_{\text{low}}$) scores $< \tau$ after the cut.
  (b) Score Drift: Non-candidate edges (previously unsuspicious) whose detector score rises $\ge \tau$ strictly after edge removal. These edges are classified as detector topology artefacts and tracked separately with their true labels rather than counted as containment failures.
- **Rationale:** In message-passing graph neural networks, pruning edges redistributes aggregation weights across neighbor nodes, which can artificially perturb the scores of unrelated background edges. Conflating detector artefacts with response failure obscures true containment effectiveness.

### Decision 45: Authentic Real-Time Windowing via FLOW_START_MILLISECONDS
- **Attribution:** Owner
- **Decision:** Utilize `FLOW_START_MILLISECONDS` timestamps present in NetFlow v3 (`NF-UNSW-NB15-v3.csv`) to construct 60-second temporal sliding windows bucketed chronologically. Maintain strict time-ordered splits (70% train, 15% validation, 15% test) across 648.66 hours of continuous network trace.
- **Rationale:** Guarantees temporal causality, prevents synthetic time leakage, and models authentic operational incident response.

### Decision 46: Stratified Unit-Test Slice Provenance & Integrity Guard
- **Attribution:** Owner
- **Decision:** Deprecate synthetic IP address slices (`192.168.1.x`). Replace `data/sample_nf_v2.csv` with an authentic stratified sample of 5,022 flows drawn directly from `NF-UNSW-NB15-v3.csv` across all 10 classes (minimum 10 samples per rare class) preserving authentic attributes and subnets. Restrict sample data strictly to fast unit tests.
- **Rationale:** Complete scientific reproducibility and strict separation between unit test testbeds and evaluation telemetry.

### Decision 47: 100-Incident Containment Benchmark on Real Test Split
- **Attribution:** Owner
- **Decision:** Execute Phase P2 evaluation across $N=100$ real incident windows from the 354,814-flow test split. Compute Attack Block Rate (ABR) and Collateral Damage Rate (CDR) with 95% bootstrap confidence intervals (1,000 resamples), generate multi-strategy continuous frontier curves across hyperparameter sweeps, and evaluate matched-ABR CDR at 80%, 90%, and 95%.
- **Rationale:** Establishes rigorous statistical confidence and demonstrates Pareto containment efficiency under genuine production conditions.

### Decision 48: Ground-Truth Fixed Incident Evaluation & Invariant Denominators
- **Attribution:** Owner
- **Decision:** Define the 100 incident evaluation sets and their regions strictly ONCE from ground truth, independent of any strategy or detection threshold $\tau$. The evaluation region $R$ is defined as the $k$-hop neighborhood ($k=2$) of all hosts involved in true attack flows within each 60-second window. All denominators ($\text{denom\_attack\_flows}$ and $\text{denom\_benign\_flows}$) are invariant constants across all strategies and sweep points.
- **Rationale:** Dynamically adjusting the collateral denominator based on which hosts are touched by cut edges creates an artificial moving target, causing wild non-monotonic fluctuations in Collateral Damage Rate (CDR). Invariant ground-truth denominators mathematically guarantee monotonicity for threshold sweeps like `BlockAllFlagged`.

### Decision 49: Unavoidable Collateral Floor & Subnet Granularity Reporting
- **Attribution:** Owner
- **Decision:** Formally define the unavoidable collateral floor as the share of benign flows sharing an `EdgeKey` (`src_ip > dst_ip : dst_port / proto`) with any true attack flow in the region. Report this per incident and overall. Furthermore, decompose blocked benign flows into `(source subnet, destination host, dst port)` tuples.
- **Rationale:** Exposes whether collateral damage stems from inherent protocol/address collision or from structural cut choices. In UNSW-NB15, attacker IPs (`175.45.176.0/24`) and benign client IPs (`59.166.0.0/24`) are disjoint, establishing an unavoidable collateral floor of 0.00% at discrete `EdgeKey` granularity.

### Decision 50: Continuous Action Space (N1+N3 Rate Limiting & Residual Risk)
- **Attribution:** Owner
- **Decision:** The containment success condition requires that every remaining edge in the post-cut graph with detector score $\ge \tau$ must be either cut (`ActionType.DROP`) or throttled (`ActionType.RATE_LIMIT`). N1's continuous optimization objective charges `DROP` at $1.0 \times \text{cost}(e)$, `RATE_LIMIT` at $\text{rate\_limit\_factor} \times \text{cost}(e)$ (default $0.3$), with a penalty for each rate-limited attack edge ($\lambda_{\text{residual\_risk}} \sum M_e \cdot \text{score}_e \cdot \text{rl\_factor}$). In evaluation metrics, rate-limited benign flows are charged $0.3 \times$ collateral damage, and Attack Block Rate (drops) and Attack Throttle Rate (rate-limits) are reported separately. `BlockAllFlagged` remains `DROP`-only.
- **Rationale:** Pure hard cuts incur binary operational penalties on sensitive services (e.g. database/DNS ports). Allowing proportional rate limiting enables the optimizer to mitigate high-cost edges with partial suppression while strictly driving malicious traffic below operational thresholds.

### Decision 51: Score Drift Re-Framing: Unmasked Attacks vs. Structural Artefacts
- **Attribution:** Owner
- **Decision:** Re-frame score drift accounting: non-candidate edges whose detector score rises $\ge \tau$ post-cut are partitioned by their ground-truth labels. Edges that are true attacks (`Label > 0`) are classified as "Unmasked attacks found by re-scoring", representing additional threat discovery enabled by graph topology pruning. Edges that are benign (`Label == 0`) are classified as "Drifted benign edges" (structural detector distortion). Both counts are reported separately for every strategy. Remove claims that zero score drift demonstrates superiority.
- **Rationale:** Pruning high-degree attack hubs unmasks latent secondary attack edges that were previously suppressed by message aggregation, which is a defensive capability rather than an artefact.

### Decision 52: Uncapped Greedy Baseline Execution
- **Attribution:** Owner
- **Decision:** Remove the hardcoded `max_cuts=50` cap in `GreedyCutSolver` (setting `max_cuts: Optional[int] = None`). Allow Greedy to cut as many candidate edges as required by its utility function without early truncation.
- **Rationale:** Eliminates artificial baseline truncation on large incident subgraphs, enabling true Pareto comparison across the full operating spectrum.

### Decision 53: TTL Feature Ablation & Non-Reliance Verification
- **Attribution:** Owner
- **Decision:** Run feature ablation benchmarking E-GraphSAGE and RandomForest on full train/val/test splits (2.36M flows) with vs without TTL features (`MIN_TTL`, `MAX_TTL`, `DNS_TTL_ANSWER`).
- **Rationale:** Verifies that detection does not hinge on dataset-specific TTL artifacts. The results confirmed negligible delta: GNN F1 shifted from $0.8905$ (with TTL) to $0.8893$ (without TTL, $\Delta = -0.0012$), and RandomForest F1 shifted from $0.9991$ to $0.9973$ ($\Delta = -0.0018$).

### Decision 54: Phase P2 Real-Data Containment Completion & Deferral of Collateral Advantage
- **Attribution:** Owner
- **Decision:** Mark Phase P2 complete on NF-UNSW-NB15-v3 with honest scientific parity: N1 achieves parity with `BlockAllFlagged` (no measured advantage on this dataset), incurring an operational latency penalty (1.03s mean vs. 188ms). Add Legitimate Collateral Damage Rate (LCDR, isolating benign flows where neither endpoint is a source of true attack) alongside label-based CDR. N1's collateral-reduction claim to be evaluated in the P3 lab with shared-infrastructure scenarios.
- **Rationale:** In NF-UNSW-NB15-v3, attacker hosts (`175.45.176.0/24`) and legitimate client subnets (`59.166.0.0/24`) are topologically disjoint; benign flows touching attacker hosts (e.g. `149.171.126.x -> 175.45.176.1:53` DNS queries) dominate label-based CDR equally across all strategies, meaning the collateral trade-off barely exists in this dataset. Honest benchmarking requires acknowledging statistical parity rather than interpolating across wide operating gaps, and evaluating topological cut advantages under genuine shared-infrastructure topology in Phase P3.

### Decision 55: Four-Stage Phase P3 Sub-Phasing (P3a / P3b / P3c / P3d)
- **Attribution:** Owner
- **Decision:** Decompose Phase P3 into four sequential sub-phases, each executed as an independent task requiring explicit owner approval:
  - **P3a**: Environment + topology + services + benign background traffic + scenario scripts + ground-truth logging + flow export. Done when a labeled flow CSV from one full run exists, with label counts per scenario and a manual check of 20 random attack and 20 random benign flows against `ground_truth.jsonl`.
  - **P3b**: Lab detector training and validation. Done when E-GraphSAGE achieves test $\text{F1} \ge 0.85$.
  - **P3c**: Containerized os-ken SDN enforcer + live N2 verification. Done when live Scenario S1 is contained and verified in $< 60$ s.
  - **P3d**: Multi-scenario comparative containment evaluation. Done when the P2-style comparison table is produced for S1–S4.
- **Rationale:** Prevents monolithic execution failure, enables incremental validation of SDN infrastructure, detector stability, and containment correctness.

### Decision 56: WSL2 Environment Verification Gate & OVS Datapath Strategy
- **Attribution:** Owner
- **Decision:** Execute `sudo modprobe openvswitch` in the target WSL2 environment before writing lab code. If the kernel module is unavailable, configure Mininet with OVS userspace datapath (`OVSSwitch datapath='user'`) and verify that OpenFlow 1.3 meters function properly. If meters fail in userspace, escalate to owner to transition to an Ubuntu VirtualBox VM.
- **Rationale:** Ensures reliable Open vSwitch datapath support for OpenFlow 1.3 meters required for rate-limiting enforcement before investing in testbed orchestration.

### Decision 57: Intra-EdgeKey Collateral Collision Design
- **Attribution:** Owner
- **Decision:** The compromised host must generate benign traffic on the exact `EdgeKey`s (`src_ip, dst_ip, dst_port, proto`) that it attacks: browsing `web:80` during S4 SYN flood, legitimate SFTP to `fileserver:22` during S3 brute force, and legitimate database queries to `db:3306` during S2 lateral movement. Both benign and attack flows are logged authoritatively. Report the count of benign flows sharing an `EdgeKey` with attack flows per scenario.
- **Rationale:** In NF-UNSW-NB15, the unavoidable collateral floor was 0.00% because attack and benign traffic were disjoint. Purposefully generating concurrent benign traffic on contested services creates authentic collateral dilemmas that test selective mitigation.

### Decision 58: Disaggregated Lab Collateral Damage Metrics
- **Attribution:** Owner
- **Decision:** In the lab environment, compute and report two distinct collateral damage rates based on script ground-truth labels (without excluding attacker hosts):
  (a) **Other-Host Collateral Damage Rate**: blocked benign flows originating from uncompromised hosts (`ws1`..`ws4`), divided by all benign flows of those hosts.
  (b) **Compromised-Host Collateral Damage Rate**: blocked benign flows originating from the compromised host, divided by all benign flows of the compromised host (rate-limited flows weighted at `rate_limit_factor`).
  Also report overall label-based CDR.
- **Rationale:** Differentiates between preserving innocent third-party enterprise services and preserving the concurrent legitimate operations of an asset under partial compromise.

### Decision 59: N2 Closed-Loop Re-Scoring Alignment
- **Attribution:** Owner
- **Decision:** Align the definition of N2 strictly with SPEC Section 6 & 12: N2 consists of re-scoring the next traffic window with the active detector post-enforcement to verify that residual malicious scores drop below detection threshold $\tau$. Active network probing is retained strictly as an auxiliary sanity check, not the core N2 metric.
- **Rationale:** Enforces detector-driven closed-loop verification, ensuring the GNN confirms containment in the live SDN data plane.

### Decision 60: Strict Train/Test Generalization Separation
- **Attribution:** Owner
- **Decision:** Enforce structural separation between training and test lab runs: the test run assigns compromise to a DIFFERENT workstation (e.g. `ws3` instead of `ws-compromised`) and modulates attack parameters (scan rates, port ranges, flood intensity, and brute-force inter-arrival timing).
- **Rationale:** Guarantees that the lab detector learns generalized graph structural anomalies rather than overfitting to specific static IP addresses or fixed temporal cadences.

### Decision 61: S2 Multi-Hop Lateral Movement Pivot Ground Truth
- **Attribution:** Owner
- **Decision:** For Scenario S2 (lateral movement from compromised host -> fileserver -> database), flows from `fileserver` to `db:3306` generated by the pivot script are explicitly logged and labeled as attack flows (`Label = 1`). Document and report that `fileserver` acts as a partial secondary attack source during S2.
- **Rationale:** Reflects authentic adversary post-exploitation behavior where compromised intermediate infrastructure generates downstream attack traffic.

### Decision 62: Gateway Single-Interface Flow Extraction (gw-eth1 Only)
- **Attribution:** Owner
- **Decision:** Run `nfstream` live capture strictly on `gw-eth1` (the server-facing interface of the central gateway), deprecating dual-interface capture. Formally document the exact feature mapping from `nfstream` attributes to the 48-feature GraphGuard schema.
- **Rationale:** Capturing on both `gw-eth0` and `gw-eth1` duplicates routed flows across the gateway. Capturing solely on `gw-eth1` observes all inter-subnet client-to-server and server-to-client traffic exactly once without artificial flow duplication.

### Decision 63: OpenFlow 1.3 Meter Band Type Standardization
- **Attribution:** Owner
- **Decision:** Standardize the OpenFlow 1.3 meter band type constant name as `OFPMBT_DROP` (OpenFlow Packet Meter Band Type Drop) in all SDN enforcer implementations and documentation, deprecating `OFPMT_DROP`.
- **Rationale:** Corrects the protocol constant naming to match OpenFlow 1.3 specification wire formats and os-ken library definitions.

### Decision 64: Disjoint Source-Port Partitioning & Exact 5-Tuple Logging
- **Attribution:** Owner
- **Decision:** Enforce non-overlapping ephemeral source-port ranges: ports `40000–44999` are strictly reserved for benign traffic generators, and ports `50000–59999` are strictly reserved for attack scenario scripts. All scripts must log their exact 5-tuples (`src_ip, dst_ip, src_port, dst_port, protocol`) with microsecond timestamps in `ground_truth.jsonl`.
- **Rationale:** Eliminates ephemeral port collision ambiguity during ground-truth labeling, enabling deterministic flow verification.

### Decision 65: Rate-Limited SYN Flood Execution (hping3)
- **Attribution:** Owner
- **Decision:** Scenario S4 must execute `hping3` with an explicit rate limit (e.g. `-i u1000` for 1,000 pkts/s, or `--faster`), strictly prohibiting unthrottled `--flood`.
- **Rationale:** Unthrottled packet flooding overwhelms virtual network namespaces, exhausts socket buffers, and hangs Mininet virtual switches, preventing realistic SDN flow measurement.

### Decision 66: Strict "Unmatched" Classification for Unverified Flows
- **Attribution:** Owner
- **Decision:** Any flow captured on `gw-eth1` that does not match an active ground-truth attack log entry AND does not match a registered benign traffic generator rule must be labeled `"unmatched"` (quarantined/flagged) rather than silently defaulted to benign (`Label = 0`).
- **Rationale:** Guarantees zero label contamination from background OS noise or stray network packets, preserving rigorous dataset integrity.

### Decision 67: Sequential Scenario Execution Schedule with 60s Quiescent Gaps
- **Attribution:** Owner
- **Decision:** Scenarios S1 through S5 must run sequentially with 60-second quiescent gaps between them during which only benign background traffic runs.
- **Rationale:** Allows active TCP connections to cleanly terminate, flushes flow exporter expiration timers, and clears GraphGuard sliding windows between scenarios to eliminate inter-scenario telemetry leakage.

### Decision 68: Per-Host Isolated Working and Service Directories
- **Attribution:** Owner
- **Decision:** Configure isolated per-host directory trees (e.g. `testbed/run/{hostname}/`) with private PID, log, and socket mount paths for all Mininet network namespace daemons (`dnsmasq`, `nginx`, `sshd`, sqlite/mysql).
- **Rationale:** Mininet hosts share the root filesystem by default; private paths prevent socket collisions, lockfile conflicts, and process interference between emulated server daemons.

### Decision 69: Pre-Flight Controller Reachability Gate
- **Attribution:** Owner
- **Decision:** Execute automated pre-flight reachability checks against the containerized os-ken controller before starting Mininet: verify TCP connection to OpenFlow port `6653` and perform an HTTP GET to `http://127.0.0.1:8080/stats/switches`.
- **Rationale:** Ensures the controller container is fully initialized and ready to accept OpenFlow 1.3 handshakes before Mininet initializes virtual bridges, avoiding silent switch disconnection failures.

### Decision 70: Ground-Truth Audit with Compounding Compromised Benign Verification
- **Attribution:** Owner
- **Decision:** The P3a acceptance audit must manually verify: (1) 20 randomly sampled attack flows, (2) 20 randomly sampled benign flows from normal hosts, and (3) **5 specific colliding benign flows from `ws-compromised`** on contested services (`web:80`, `fileserver:22`, `db:3306`), confirming 100% ground-truth label agreement.
- **Rationale:** Empirically verifies that benign business traffic generated by the compromised asset on contested ports is not falsely labeled as malicious.

### Decision 71: S4 Source-Port Restraint (hping3 Rate and Baseport)
- **Attribution:** Owner
- **Decision:** Scenario S4 executes `hping3 -S -p 80 -i u10000 --baseport 50000` for 60 s (100 pkts/s, ports 50000–55999, staying strictly within the attack range). Log the actual source-port range used. Never permit attack ports to leave 50000–59999.
- **Rationale:** Constrains SYN flood source ports strictly to the allocated attack port partition, eliminating port boundary overlap and packet queue saturation.

### Decision 72: Empirical NFStream Feature Mapping & Dynamic Constant Drop
- **Attribution:** Owner
- **Decision:** Before building the flow exporter, run `NFStreamer(statistical_analysis=True)` on a short capture and print the real column list. Rebuild the mapping table using only attributes that genuinely exist in that list. Strictly no default-filled dummy features for attributes NFStream does not provide. After the run, drop any feature that is constant across all flows. Report the final real feature count and CSV column count accordingly.
- **Rationale:** Eliminates fabricated zero-filled or default columns, ensuring the GNN receives authentic empirical telemetry generated by the flow probe.

### Decision 73: Dual-Branch OpenFlow 1.3 Meter Verification in check_env.sh
- **Attribution:** Owner
- **Decision:** `check_env.sh` must start `openvswitch-switch` and verify OpenFlow 1.3 meter functionality (`OFPMBT_DROP`) in BOTH branches: kernel datapath and userspace (`netdev`) datapath.
- **Rationale:** Validates meter support across both kernel module and user-space OVS configurations, ensuring rate-limiting enforcer viability before running Mininet experiments.

### Decision 74: Python-Bound Benign Generators in Port Range 40000–44999
- **Attribution:** Owner
- **Decision:** All benign background traffic generators must be Python clients that explicitly bind an ephemeral source port in `40000–44999` (`paramiko` for SSH/SFTP, Python SMB client, `requests`/`http.client` with bound socket, `dnspython` for DNS, and Python MySQL client). Do not use CLI tools that cannot bind source ports.
- **Rationale:** Guarantees strict non-overlapping source port separation between benign and attack traffic, preventing OS ephemeral port allocation leakage into attack ranges.

### Decision 75: Exact EdgeKey Collision Counting
- **Attribution:** Owner
- **Decision:** Same-EdgeKey collision count strictly uses the exact canonical `EdgeKey` (`src_ip, dst_ip, dst_port, proto`).
- **Rationale:** Unifies collision measurement under GraphGuard's fundamental multi-graph edge identity model.


