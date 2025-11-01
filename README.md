GeoFL — Evaluating Failover Strategies in Geo-Distributed Federated Learning

This repository contains three evaluation scripts that benchmark three orchestration strategies for Federated Learning (FL) under three progressive failure rates. The goal is to assess resilience and model quality when regional aggregators experience outages or severe degradation.

What’s inside

f2.py — experiments with Light Stress (≈ 20% failed rounds)

f5.py — experiments with Moderate Stress (≈ 50% failed rounds)

f8.py — experiments with Extreme Stress (≈ 80% failed rounds)

Each script runs MNIST on a geo-distributed setup with two regions and client splits that induce natural non-IID data. Results are written to results/ as CSV and JSON for side-by-side comparison.

Strategies under test

Centralized FL — a single parameter server coordinates all rounds.

Hierarchical FL — regional supernodes pre-aggregate, then forward to a global coordinator.

GeoFL (Agentic Failover) — clients can reassociate across regions via multi-criteria failover (latency, queueing, reliability, checkpoint freshness) with deduplication and progress-preserving checkpoints.

These three strategies let you compare “stop-and-wait” behavior (centralized), compartmentalized resilience (hierarchical), and quality-aware mobility (GeoFL).

Quick start
1) Environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # torch/torchvision, flwr, numpy, pandas, matplotlib, etc.


The scripts auto-download MNIST (fallback mirrors included). If your environment blocks downloads, place MNIST files under ./data/MNIST/processed/.

2) Run experiments

Each script accepts a --strategy flag with one of: centralized, hierarchical, geofl.

Light Stress (~20% failures)
python f2.py --strategy centralized
python f2.py --strategy hierarchical
python f2.py --strategy geofl

Moderate Stress (~50% failures)
python f5.py --strategy centralized
python f5.py --strategy hierarchical
python f5.py --strategy geofl

Extreme Stress (~80% failures)
python f8.py --strategy centralized
python f8.py --strategy hierarchical
python f8.py --strategy geofl


Tip: If you prefer to run all strategies for a scenario in one go, use --strategy all (if supported in your build). Otherwise, invoke each command separately as above.

Outputs

After each run you’ll find:

results/<scenario>/<strategy>_rounds.csv — per-round metrics (loss, accuracy, participation, etc.)

results/<scenario>/<strategy>_rounds.json — summary and configuration

Optional plots: results/<scenario>/*acc*.png, *loss*.png (if plotting is enabled)

Example rows in the CSV:

round	global_val_acc	global_val_loss	participated_clients	regionA_queue	regionB_queue	fail_flag

This structure makes it easy to compute:

Resilience rate (operational rounds / total rounds)

Final loss/accuracy

Recovery latency (rounds to return to pre-failure quality)

Participation balance / fairness (per-region contribution share)

Duplicate-avoid rate (if deduplication logging is enabled)

Reproducibility

Fixed seeds for data splits and client sampling

Deterministic failure schedules per scenario (see each script’s header)

All configs logged to the JSON summary alongside metrics

Interpreting results (at a glance)

Centralized FL: simple but sensitive; accuracy and participation collapse as failures increase.

Hierarchical FL: continues operating under regional outages but can slowly erode model quality when a region stays dark.

GeoFL: maintains operation and typically achieves equal or better final loss, with faster recovery and fewer oscillations thanks to quality-aware reassociation and checkpoint deduplication.
