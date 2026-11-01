# TabFPA: Tabular Feasibility-Preserving Attack

TabFPA is a feasibility-preserving adversarial attack for heterogeneous tabular
data. It compiles the declared domain constraints (immutable fields, integer
domains, linear/nonlinear relations, derived features) into the *shape* of the
action space itself, so that every candidate proposed to the victim model is a
complete, constraint-valid point by construction: joint discrete moves are
sampled through ST-Gumbel estimators, continuous displacements are repaired by
dataset-specific conditional completions, and gradients are bridged back to the
raw inputs with BPDA. A greedy ratchet keeps the best certified flip under a
strict query budget.

This repository contains the implementation and the reproduction entries for the
main experiments of the paper: attacks on standard (default) and
adversarially-trained (Madry) targets under an L2 budget, plus two additional
constraint scenarios (Malware, and WiDS with an added diagnosis–body-system
mapping relation).

## Repository layout

| Path | Contents |
|---|---|
| `method/core/` | Constraint compilation (`feasible_integer_moves`), conditional completion base, joint-move scoring/judging, WiDS completions and the added mapping relation |
| `method/scripts/` | The released joint discrete–continuous search loop (`feasible_gumbel_ablation`) and its dataset loaders/utilities |
| `method/tabdatasets/` | Per-dataset feasible maps (URL, WiDS, LCLD, Malware, CTU) |
| `method/models/` | Victim-model adapters (raw-feature interface, transforms built in) |
| `method/widen_kernel_*.py` | Search schedules: `w_reinit` (main), `query200`, `temperature_params` (Madry URL), `ballmap_dir` (kernel superset used by the CTU entry) |
| `method/official_defaults.{py,json}` | Frozen per-scenario parameter profiles |
| `method/lcld_*` | LCLD frozen 996 configuration and official-draw pool builder |
| `method/wids_mapping_*.py` | WiDS mapping-constrained scenario |
| `full_wreinit_run.py`, `ctu_dlr03_run.py`, `run_official.py`, `run_experiments.py` | Experiment entry points (see below) |
| `run_*.sh` | Launchers (relative paths only; `PYTHON` and `GPU` overridable) |
| `tabularbench/` | Vendored [TabularBench](https://github.com/serval-uni-lu/tabularbench) package (model wrappers, constraint checkers, dataset factory) |

## Environment

- Python 3.8–3.10 with PyTorch 1.11/1.12 and **a CUDA GPU** — the main entries
  load victim models on CUDA (a CUDA device is required; CPU-only execution is
  not supported by the experiment entries).
- Install the Python dependencies into your environment, e.g.
  `pip install -r requirements.txt` (this includes the vendored benchmark's
  pinned dependencies plus `huggingface_hub` for the dataset fallback).
- Every launcher accepts `PYTHON=/path/to/python` and `GPU=<cuda index>`
  environment overrides and writes outputs under `./runs/` (relative to the
  repository root).

## Datasets and target models

The attack runs against the public TabularBench assets (Simonetto, Ghamizi,
Cordy; arXiv:2408.07579, also used by the CAA attack paper arXiv:2406.00775).

1. Obtain the datasets and the pretrained target-model checkpoints from the
   official [TabularBench release](https://github.com/serval-uni-lu/tabularbench).
2. Place them as (or symlink to):
   - `tabularbench/datasets/` (dataset files), and
   - `tabularbench/target_model/` (checkpoints, e.g.
     `target_model/url/tabnet_url_default.model`, `..._madry.model`).
   
   Alternatively point `TABULARBENCH_DATASET_DIR` / `TABULARBENCH_TARGET_MODEL_DIR`
   at existing locations.

Victim models are the five TabularBench architectures — TabTransformer, RLN,
VIME, STG, TabNet — with `_default` (standard) and `_madry` (adversarially
trained) checkpoints. The wrappers embed their input transforms: always feed
**raw** features.

## Running the experiments

All entries default to seeds 0–4, batch 128, 200 search steps, and a hard cap of
256 forward evaluations per sample (the WiDS default entry uses a 200-query
cap); per-batch seeds are `seed*1000003 + offset`. Pools are the legal
constraint-valid class-1 test rows, filtered to per-model clean-correct
predictions, original order — except LCLD, which attacks the clean-correct
subset of the official stratified 1000-row draw (`random_state=42`).

### 1. Standard targets under L2 (default checkpoints)

```bash
# URL
bash run_full_wreinit.sh --dataset url --model tabnet_default     # × 5 models
# CTU (per-cell d_lr=0.03 tuning)
bash run_ctu_dlr03.sh --model tabnet_default                      # × 5 models
# LCLD (frozen 996 configuration: build the draw pools once, then one model per run)
bash run_lcld_996.sh pools
bash run_lcld_996.sh run --model tabnet_default                   # × 5 models
# WiDS (frozen profile wids_d09_h32_lr36)
bash run_official.sh --dataset wids
```

### 2. Adversarially trained targets (Madry checkpoints, L2)

```bash
# URL (frozen profile url_madry_k30_lr70_tau05, all five models in one call)
bash run_official.sh --dataset url --models tabtransformer_madry torchrln_madry vime_madry stg_madry tabnet_madry

# WiDS (standard w_reinit schedule) — repeat for each of the five *_madry models
bash run_full_wreinit.sh --dataset wids --model tabtransformer_madry

# CTU (standard w_reinit schedule) — repeat for each of the five *_madry models
bash run_full_wreinit.sh --dataset ctu_13_neris --model tabtransformer_madry

# LCLD (996 configuration on the official draw; the pools step above already
# builds the Madry pools) — four Madry models; tabnet_madry has an empty
# clean-correct pool and is pool-only
bash run_lcld_996.sh run --model tabtransformer_madry
```

### 3. Additional scenario: Malware (L2)

```bash
bash run_full_wreinit.sh --dataset malware --model tabnet_default # × 5 models
```

### 4. Additional scenario: WiDS with added mapping relation (L2, radius 0.6)

```bash
bash run_wids_mapping.sh        # all five default models, seeds 0–4
```

### Outputs

- `run_full_wreinit.sh` / `run_ctu_dlr03.sh`: `runs/<dataset>_full_wreinit/<model>/`
  and `runs/ctu_dlr03_full/<model>/` — `manifest.json`, `status.json`, resumable
  `state.sqlite3` (per-seed scene rows `S/N/D/RA`; adversarial batches as BLOBs).
- `run_official.sh`: `runs/official_<dataset>_<profile>_q200/<model>/` — per-seed
  results plus `complete.json` with the frozen parameters.
- `run_lcld_996.sh`: pools under `reference/caa_pools/`, results under
  `runs/lcld996_f0draw_20260930/` (JSON + per-seed NPZ).
- `run_wids_mapping.sh`: `runs/wids_mapping_full_wreinit_eps06/<model>/`
  (`results.json` rows with `S`, `N`, `D`, `RA`).

Metrics: per seed, `S` = certified constraint-valid misclassifications among the
`N` attacked (clean-correct, legal) rows out of the `D` eligible pool rows;
RA = (N − S)/D and ASR = S/N. Report five-seed mean ± SD.

## Protocol notes

- The L2 radius is 0.5 in the min-max transformed space (0.6 for the WiDS
  mapping scenario; internal completion radius 0.495).
- The LCLD entry sets `WK_AUDIT_TOL=1e-12` internally (final audit tolerance for
  the float64 recomputation of the derived-equality columns); do not override.
- Victims are evaluated in eval mode with the official pretrained weights; no
  retraining is performed.
- Query accounting: every accepted candidate passes a counted float64
  certification forward pass; a free screen (misclassified ∧ within budget)
  does not consume the query budget.

## Provenance

Code is frozen from the research codebase at paper-submission state; only the
main-experiment entries are included (ablation and diagnostic harnesses are not
part of this release). The LCLD tuning module pins the kernel hash
(`EXPECTED_KERNEL` in `method/lcld_tune100.py`) and verifies
`method/lcld_schedule_kernel.py` byte-identity at run time.

## License

`tabularbench/` is the vendored upstream TabularBench package under its MIT
license (`tabularbench/LICENSE`, © 2024 Thibault Simonetto, Salah Ghamizi,
Maxime Cordy). The TabFPA code in this repository is released for
reproducibility of the paper.
