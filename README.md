# DriftLA

**Drift-aware streaming graph recommendation**

This repository contains the implementation, experiment artifacts, and reproduction scripts for DriftLA.

---

## Quick start

Two paths are supported: **verify bundled numbers without data**, or **rerun from scratch** after downloading datasets.

### A. Verify bundled results (no datasets, ~1 minute)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/verify_results.py
```

This prints headline Recall@10 / NDCG@10 (mean±std over seeds 42–46) from the bundled `results/` JSON logs and checks the 3×3 streaming protocol. Optional strict check:

```bash
python3 scripts/verify_results.py --require-all-seeds
```

Regenerate LaTeX table fragments from the same JSONs:

```bash
python3 scripts/make_tables.py
```

### B. Run from scratch (GPU recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src"

# 1) Install at least MovieLens-1M (see data/README.md)
mkdir -p data && cd data
curl -L -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip && rm ml-1m.zip
cd ..

# 2) Smoke test (CPU OK, ~1–2 min)
python3 -m driftla.train --data_root data --dataset ml-1m --device cpu --smoke

# 3) Save bundled reference, then rerun one headline experiment (CUDA, ~1–2 h on ML-1M)
cp results/driftla/driftla_valrouted_3x3_ml1m_seed42.json /tmp/bundled.json
python3 scripts/run_single.py valrouted ml-1m --seed 42 --force

# 4) Compare rerun to bundled numbers
python3 scripts/verify_results.py \
  --compare results/driftla/driftla_valrouted_3x3_ml1m_seed42.json \
  --reference /tmp/bundled.json
```

Download links for all benchmarks, expected folder layout, and per-dataset commands are in [data/README.md](data/README.md).

| Goal | Command |
|------|---------|
| Dense headline DriftLA (val-routed) | `python3 scripts/run_single.py valrouted ml-1m --seed 42` |
| DriftLA-Adapter | `python3 scripts/run_single.py adapter ml-1m --seed 42` |
| Sparse / negative DriftLA (champion) | `python3 scripts/run_single.py champion gowala --seed 42` |
| LightGCN-WS baseline | `python3 scripts/run_single.py lightgcn_ws ml-1m --seed 42` |
| Leave-one-out ablation | `python3 scripts/run_ablations.py --datasets ml-1m ciao` |

All full experiments use **3 warmup epochs + 3 streaming passes** and seeds **42–46** in the bundled logs. Existing valid JSON files are skipped unless `--force` is passed.

---

## Repository layout

```
driftla/
├── README.md                 Reproduction guide (this file)
├── requirements.txt          Python dependencies
│
├── src/                      Implementation (set PYTHONPATH here)
│   ├── driftla/              DriftLA model and training CLI
│   └── baselines/            Streaming baselines (LightGCN-WS, PECL, SPMF, SimGCL-WS, ...)
│
├── results/                  Experiment JSON logs (included for reproduction)
│   ├── driftla/
│   └── baselines/
│
├── data/                     Dataset folder (empty; download instructions in data/README.md)
│
└── scripts/
    ├── verify_results.py     Summarize bundled JSON logs (no data required)
    ├── make_tables.py        Optional: LaTeX table fragments from JSON logs
    ├── run_single.py         Run one experiment with paper defaults
    ├── run_ablations.py      Batch leave-one-out ablation runs
    ├── experiment_utils.py   Shared experiment helpers
    ├── streaming_protocol.py Streaming evaluation protocol helpers
    └── data/
        └── materialize_dense_kcore.py   Optional dense dataset export
```

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 or later |
| PyTorch | 2.0 or later |
| GPU | Optional; CPU smoke tests supported |

---

## Installation

```bash
cd driftla
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src"
```

Download benchmark datasets into `data/` following [data/README.md](data/README.md). **Raw data files are not included in this repository.** Use the provider links in that guide to download MovieLens, Ciao, Gowalla, Yelp, and Amazon Reviews 2023.

---

## Dataset acquisition

This repository includes source code and experiment JSON logs (`results/`). **Benchmark datasets are not uploaded.** To re-run training, download each dataset from its original provider and place it under `data/` as documented in [data/README.md](data/README.md).

| Dataset | Paper name | Download |
|---------|------------|----------|
| MovieLens-1M | ML-1M | [GroupLens](https://grouplens.org/datasets/movielens/1m/) · [ZIP](https://files.grouplens.org/datasets/movielens/ml-1m.zip) |
| Ciao | Ciao | [UIC](https://www.cs.uic.edu/~liub/CA/research.html) · [RAR](https://www.cs.uic.edu/~liub/CA/rating.rar) |
| MovieLens-10M | ML-10M-cap | [GroupLens](https://grouplens.org/datasets/movielens/10m/) · [ZIP](https://files.grouplens.org/datasets/movielens/ml-10m.zip) |
| MovieLens-20M | ML-20M-cap | [GroupLens](https://grouplens.org/datasets/movielens/20m/) · [ZIP](https://files.grouplens.org/datasets/movielens/ml-20m.zip) |
| Gowalla | Gowalla 5-core | [SNAP](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.html) · [TXT.GZ](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz) |
| Yelp Academic | Yelp 5-core | [Yelp Open Dataset](https://www.yelp.com/dataset) (registration required) |
| Amazon Reviews 2023 | Amazon23-Magazine | [Project page](https://amazon-reviews-2023.github.io/) · [GitHub](https://github.com/McAuley-Lab/Amazon-Reviews-2023) |

Full placement instructions, example shell commands, and expected folder layout are in [data/README.md](data/README.md).

---

## Running experiments

All experiments use five random seeds: **42, 43, 44, 45, 46**. Outputs are written as JSON files under `results/driftla/` or `results/baselines/`.

### DriftLA (dense benchmarks: validation-only routing)

```bash
export PYTHONPATH="${PWD}/src"
python3 -m driftla.train \
  --data_root data \
  --dataset ml-1m \
  --preset v3_champion \
  --use_dataset_preset \
  --routing auto_density_drift \
  --val_tail_frac 0.2 \
  --warmup_epochs 3 \
  --streaming_passes 3 \
  --seed 42 \
  --device cuda \
  --out_json results/driftla/driftla_valrouted_3x3_ml1m_seed42.json
```

Equivalent wrapper: `python3 scripts/run_single.py valrouted ml-1m --seed 42`

### Baselines

```bash
python3 -m baselines \
  --method lightgcn_ws \
  --dataset ml-1m \
  --data_root data \
  --seed 42 \
  --device cuda
```

Supported baseline methods: `lightgcn_ws`, `simgcl_ws`, `pecl`, `spmf`, `lightgcn_window`, `graphsail_ws`, `ergnn_ws`.

### Convenience wrappers

```bash
python3 scripts/run_single.py valrouted ml-1m --seed 42
python3 scripts/run_single.py lightgcn_ws ciao --seed 42
python3 scripts/run_ablations.py --datasets ml-1m ciao
```

---

## Experiment JSON naming

JSON files follow these patterns (with `{dataset}` and `{seed}` placeholders):

| Role | Example path |
|------|--------------|
| Val-routed DriftLA (dense benchmarks) | `results/driftla/driftla_valrouted_3x3_{dataset}_seed{seed}.json` |
| DriftLA champion / sparse runs | `results/driftla/driftla_v3_champion_3x3_{dataset}_seed{seed}.json` |
| DriftLA-Adapter | `results/driftla/driftla_adapter_3x3_{dataset}_seed{seed}.json` |
| Leave-one-out ablations | `results/driftla/driftla_champion_ablation_{ab_id}_3x3_{dataset}_seed{seed}.json` |
| Gate controls (fixed preset) | `results/driftla/driftla_champion_gate_{mode}_3x3_{dataset}_seed{seed}.json` |
| Gate controls (val-routed) | `results/driftla/driftla_valrouted_gate_{mode}_3x3_{dataset}_seed{seed}.json` |
| Gamma trajectory logging | `results/driftla/driftla_gammalog_3x3_{dataset}_seed{seed}.json` |
| Baselines (3x3 protocol) | `results/baselines/{method}_3x3_{dataset}_seed{seed}.json` |
| SimGCL val-tuned | `results/baselines/simgcl_ws_valtuned_3x3_{dataset}_seed{seed}.json` |
| Compute-matched baselines | `results/baselines/{method}_computematched_3x3_{dataset}_seed{seed}.json` |
| Symmetric holdout baselines | `results/baselines/{method}_valhold_3x3_{dataset}_seed{seed}.json` |
| Amazon23 negative result | `results/driftla/driftla_v3_champion_3x3_amazon23_Magazine_Subscriptions_k2_seed{seed}.json` |

Dense benchmark datasets: `ml1m`, `ciao`, `ml10m_cap300k`, `ml20m_cap300k`. Sparse / negative datasets: `gowala`, `yelp`, `amazon23_Magazine_Subscriptions_k2`.

---

## Result provenance

### Figures

| Figure | Underlying data (for reference only) |
|--------|--------------------------------------|
| Architecture (Fig. 1) | Method schematic (manual layout) |
| Ablation impact (Fig. 2) | Ablation JSONs vs. full DriftLA champion on dense datasets |
| Drift-width schedule (Supp.) | Drift score to path-width mapping |
| Accuracy vs. latency (Supp.) | Recall@10 and per-batch latency from dense-benchmark JSONs |
| Cumulative lift (Supp.) | Per-batch Recall@10: val-routed DriftLA vs. LightGCN-WS |
| Loss trajectory (Supp.) | Per-batch training loss from DriftLA and PECL JSONs |
| Gamma trajectory (Supp.) | Per-batch drift gate from `driftla_gammalog_3x3_*` JSONs |

Underlying trajectories aggregate seeds **42–46** where applicable.

### Tables

| Table | Label | Source data |
|-------|-------|-------------|
| Accuracy summary (R@10 / N@10) | `tab:accuracy_summary` | `driftla_valrouted_3x3_*`, `lightgcn_ws_3x3_*`, `simgcl_ws_valtuned_3x3_*`, `pecl_3x3_*`, `spmf_3x3_*` on dense datasets |
| Per-batch latency | `tab:per_batch_latency` | Per-batch `timings.total_s` from the same dense-benchmark JSONs |
| Symmetric holdout | `tab:symmetric_holdout` | Val-routed DriftLA JSONs vs. `*_valhold_3x3_*` and `lightgcn_window_valhold_3x3_*` baselines |
| Val-routed gate controls | `tab:valrouted_gate_recall` | `driftla_valrouted_gate_{drift,ungated,fixed,plain_lora}_3x3_*` JSONs |
| Ablation (Recall@10) | `tab:ablation_recall` | `driftla_champion_ablation_{ab_id}_3x3_*` JSONs vs. full DriftLA champion |
| ML-10M cap sensitivity | `tab:cap_sensitivity` | `*_3x3_ml10m_cap{300k,1m,3m}_seed*` JSONs for DriftLA, LightGCN-WS, SimGCL-WS |
| Compute-matched baselines | `tab:computematched` | Val-routed DriftLA vs. `*_computematched_3x3_*` and default baseline JSONs |
| DriftLA-Adapter results | `tab:adapter_results` | `driftla_adapter_3x3_*`, `driftla_v3_champion_3x3_*`, `lightgcn_ws_3x3_*` JSONs |
| Negative / sparse results | `tab:negative_results` | Champion DriftLA and baseline JSONs on `gowala`, `yelp`, `amazon23_Magazine_Subscriptions_k2` |
| Ablation (NDCG@10) | `tab:ablation_ndcg` | Same ablation JSONs as `tab:ablation_recall` |
| Gate ablation (Recall / NDCG) | `tab:gate_ablation_recall`, `tab:gate_ablation_ndcg` | `driftla_champion_gate_*_3x3_*` JSONs |
| Significance (DriftLA vs. baselines) | `tab:significance` | Paired seed-level Recall@10 from val-routed DriftLA vs. LightGCN-WS, SimGCL-WS, PECL |
| Adapter significance | `tab:adapter_significance` | `driftla_adapter_3x3_*` vs. baseline JSONs |
| Matched warmup comparison | `tab:matched_warmup` | Matched-warmup JSON variants for DriftLA, adapter, and baselines |
| Continual baselines | `tab:continual_baselines` | Val-routed DriftLA vs. `lightgcn_window`, `graphsail_ws`, `ergnn_ws` JSONs |
| Amazon23 negative result | `tab:amazon_negative` | Amazon23 Magazine Subscriptions JSONs for DriftLA and baselines |

---

## Evaluation protocol

Experiments follow a **3 warmup epochs + 3 streaming passes** prequential protocol with chronological batching. Dense benchmarks use validation-only routing for DriftLA (80% warmup prefix, 20% held out for routing calibration).

---

## Support

For dataset downloads, folder layout, and example commands, see [data/README.md](data/README.md).

---

## License

Source code and experiment artifacts are provided for research reproducibility. Dataset licenses remain with their original providers (GroupLens, SNAP, Yelp, Amazon Reviews 2023, etc.).
