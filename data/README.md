# GraphGuard Datasets

This directory contains NetFlow network traffic datasets used for training, evaluating, and testing GraphGuard.

## 1. Bundled Sample Dataset (`sample_nf_v2.csv`)
- **Location:** `data/sample_nf_v2.csv` (tracked in repository)
- **Format:** NetFlow v2/v3 feature representation (54 features including `FLOW_START_MILLISECONDS`, IP addresses, Layer 4 ports, throughput, and packet distributions).
- **Purpose:** Enables instant out-of-the-box unit testing, demonstration runs, model loading, and counterfactual minimal cut containment verification without requiring hundreds of megabytes of raw downloads.
- **Provenance:** Stratified sample of 5,022 authentic flows sampled across all attack categories (Benign, DoS, Exploits, Generic, Reconnaissance, Fuzzers, Backdoor, Analysis, Worms, Shellcode) from the NF-UNSW-NB15 benchmark.

## 2. Full Benchmark Datasets (Optional for Large-Scale Training)
To train or evaluate on the complete multi-gigabyte datasets:
1. Download **NF-UNSW-NB15-v2** or **NF-UNSW-NB15-v3** CSV:
   - [NetFlow Datasets (University of Queensland / Kaggle)](https://staff.itee.uq.edu.au/mwr/datasets.html)
2. Place the CSV file in `data/raw/` (e.g., `data/raw/NF-UNSW-NB15-v3.csv`).
3. Note: `data/raw/` is intentionally listed in `.gitignore` to avoid checking large multi-hundred-megabyte files into git history.
4. Update `config/detection.yaml` if needed:
   ```yaml
   data:
     raw_path: "data/raw/NF-UNSW-NB15-v3.csv"
     sample_path: "data/sample_nf_v2.csv"
   ```
