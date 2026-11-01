# TabFPA: Tabular Feasibility-Preserving Attack

TabFPA studies feasibility-preserving adversarial attacks on heterogeneous tabular data. This release contains only the following TabFPA experiments under an L2 perturbation budget.

**Status: Under Preparation.** The scoped method code and experiment entries are available. Full experiment validation and reproduction documentation are still being finalized.

## Released experiments

| Dataset or scenario | Checkpoints | Entry |
|---|---|---|
| URL | default | `run_full_wreinit.sh` |
| URL | Madry | `run_official.sh` |
| WiDS | default | `run_official.sh` |
| WiDS | Madry | `run_full_wreinit.sh` |
| CTU (`ctu_13_neris`) | default | `run_ctu_dlr03.sh` |
| CTU (`ctu_13_neris`) | Madry | `run_full_wreinit.sh` |
| LCLD (`lcld_v2_iid`) | default and Madry | `run_lcld_996.sh` |
| WiDS-mapping | default | `run_wids_mapping.sh` |
| Malware | default | `run_full_wreinit.sh` |

The repository contains the scoped TabFPA attack entries and training code for two additional tree targets (XGBoost and LightGBM), described below. Historical tuning, ablation and baseline-comparison drivers and precomputed result tables are not included. TabularBench is an external dependency. Its source code, baseline attacks, training scripts, datasets and model weights are not included in this repository.

## Environment and assets

- Linux, Python 3.8–3.10, PyTorch 1.11/1.12 and a CUDA GPU. The main entries require CUDA. The Malware entry loads the full HDF5 table and needs sufficient host memory for multiple DataFrame and tensor copies.
- Obtain the external source, datasets and pretrained targets from the official [TabularBench repository](https://github.com/serval-uni-lu/tabularbench#data-availability). Follow its data-availability instructions for the download locations and directory structure. TabFPA does not redistribute or retrain these targets.
- The external API is checked against TabularBench commit `bfb75415a6a31a41ddfeef34478eea1da227d19c`. Keep this checkout outside TabFPA, for example as a sibling directory:

```bash
# Run from the TabFPA repository root, in a compatible Python environment.
git clone https://github.com/serval-uni-lu/tabularbench.git ../tabularbench
git -C ../tabularbench checkout bfb75415a6a31a41ddfeef34478eea1da227d19c
export TABULARBENCH_ROOT="$(cd ../tabularbench && pwd)"
pip install -r "$TABULARBENCH_ROOT/requirements.txt"
pip install -r requirements.txt
# Set these to the assets obtained following the upstream instructions.
export TABULARBENCH_DATASET_DIR=/absolute/path/to/datasets
export TABULARBENCH_TARGET_MODEL_DIR=/absolute/path/to/target_model
```

Expected examples are `$TABULARBENCH_DATASET_DIR/wids/wids.csv`, `wids/wids_metadata.csv`, and `$TABULARBENCH_TARGET_MODEL_DIR/url/tabnet_url_default.model/`. Dataset subdirectories are `url`, `wids`, `ctu_13_neris`, `lcld_v2` and `malware`; the LCLD checkpoint subdirectory is `lcld_v2_iid`. The path adapter uses upstream dataset definitions, splits, constraints, scalers and model classes without modifying the external checkout.

- Run commands from the repository root. Launchers accept `PYTHON=/path/to/python` and `GPU=<cuda index>`. They run offline, so all requested datasets and checkpoints must be available locally beforehand.

Target architectures are `tabtransformer`, `torchrln`, `vime`, `stg` and `tabnet`; append `_default` or `_madry`. Model wrappers include their transforms and consume raw features. Repeat single-model examples for the applicable architectures. Default seeds are 0–4.

## Default targets

```bash
# URL
bash run_full_wreinit.sh --dataset url --model tabnet_default
# CTU
bash run_ctu_dlr03.sh --model tabnet_default
# WiDS: all five default targets
bash run_official.sh --dataset wids
# LCLD: prepare default and Madry pools, then run one target
bash run_lcld_996.sh pools
bash run_lcld_996.sh run --model tabnet_default
```

## Madry targets

```bash
# URL: all five Madry targets
bash run_official.sh --dataset url --models tabtransformer_madry torchrln_madry vime_madry stg_madry tabnet_madry
# WiDS and CTU: repeat for the five Madry targets
bash run_full_wreinit.sh --dataset wids --model tabtransformer_madry
bash run_full_wreinit.sh --dataset ctu_13_neris --model tabtransformer_madry
# LCLD: use pools prepared above
bash run_lcld_996.sh run --model tabtransformer_madry
```

The LCLD attack entry supports `tabtransformer_madry`, `torchrln_madry`, `vime_madry` and `stg_madry`. For the original released checkpoint, LCLD `tabnet_madry` has an empty clean-correct draw: the pool builder includes it for provenance, but the attack entry excludes it.

## Additional scenarios

```bash
# Malware: repeat for the five default targets
bash run_full_wreinit.sh --dataset malware --model tabnet_default
# WiDS-mapping: all five default targets, L2 radius 0.6
bash run_wids_mapping.sh
```

The frozen diagnosis-to-body-system mapping is included in `method/config/wids_apache3j_intervals.json`. This is constraint configuration, not a dataset or a result table.

## Additional tree-target training

`targets/train_tree_targets.py` trains **XGBoost and LightGBM** as additional non-differentiable targets for transfer evaluation. It uses the existing external TabularBench datasets; the neural target checkpoints above are not needed for this training step. Dataset splits follow TabularBench: fit on `train`, use `val` for early stopping, and report clean accuracy and ROC AUC on `test`. The training seed defaults to 0 and both models use eight CPU threads.

After preparing the base environment and dataset paths above:

```bash
pip install -r targets/requirements.txt
# Train both tree targets on one dataset.
bash run_tree_targets.sh --dataset url
# The source training code also provides an optional class-weighted setting.
bash run_tree_targets.sh --dataset url --class-weight
```

Dataset choices are `url`, `wids`, `ctu_13_neris`, `lcld_v2_iid` and `malware`. Repeat the command for the desired datasets. `PYTHON=/path/to/python` selects the interpreter; this entry trains on CPU and does not require a GPU. `--out /absolute/path/to/output` changes the output root. An existing run directory is never overwritten.

| Setting | XGBoost | LightGBM |
|---|---|---|
| Maximum boosting rounds | 4000 | 2000 |
| Learning rate | 0.05 | 0.05 |
| Tree configuration | depth 8, histogram method | 64 leaves, row-wise |
| Subsample / column fraction parameters | 0.8 / 0.8 | 0.8 / 0.8 |
| Early-stopping patience | 100 | 100 |
| Optional `--class-weight` | training negatives / positives | `balanced` |

The source settings are retained, including LightGBM's default `subsample_freq=0` (its row-subsample parameter is therefore inactive). Raw features are converted to float32, with categorical codes kept as individual columns. Malware is read in chunks into a sparse CSR matrix. This entry does not fit a min-max scaler or one-hot encoder.

Local outputs are written to `runs/tree_targets/<dataset>/seed<seed>/<unweighted-or-balanced>/`: `xgb.json`, `lgbm.txt` and `manifest.json`. The manifest records actual estimator parameters, versions, feature order, data and split hashes, best iterations and test metrics. Saved models are reloaded to check test-prediction equality. `load_targets(directory)` and `predict_one(model, X)` in the training module provide loading and binary-prediction helpers; preserve the recorded feature order and input representation.

The optional dependencies are pinned to versions recorded in the source experiments. This entry provides target training and loading; transfer-attack evaluation is not included in this addition. Generated models, datasets and result files remain local and are not distributed in this repository.

## Protocol and outputs

- The L2 radius is 0.5 in min-max transformed coordinates; WiDS-mapping uses 0.6. Search uses 200 steps and batches of 128. Per-batch seeds are `seed * 1000003 + offset`.
- The attack-query cap is 256 per sample, except `run_official.sh`, which uses 200 for WiDS default and URL Madry. Initial clean evaluations and final offline audits are excluded from the attack-query count.
- Pools use constraint-valid class-1 test rows, then retain per-model clean-correct predictions in original order. LCLD uses the clean-correct subset of the official stratified 1000-row draw (`random_state=42`).
- The LCLD entry fixes `WK_AUDIT_TOL=1e-12` for the final equality audit. Frozen parameters are in `method/lcld_config.py` and `method/official_defaults.json`; no parameter search is performed.

| Entry | Output location |
|---|---|
| `run_full_wreinit.sh` | `runs/<dataset>_full_wreinit/<model>/` |
| `run_ctu_dlr03.sh` | `runs/ctu_dlr03_full/<model>/` |
| `run_official.sh` | `runs/official_<dataset>_<profile>_q200/<model>/` |
| `run_lcld_996.sh pools` | `reference/caa_pools/` |
| `run_lcld_996.sh run` | `runs/lcld996_f0draw_20260930/` or `runs/lcld996_f0draw_madry/` |
| `run_wids_mapping.sh` | `runs/wids_mapping_full_wreinit_eps06/<model>/` |

Full-pool and CTU entries store progress and adversarial batches in `state.sqlite3`, plus `manifest.json` and `status.json`; per-seed summaries are printed to standard output. Official entries write per-seed JSON/NPZ and `complete.json`. LCLD writes per-seed JSON/NPZ and verification summaries. WiDS-mapping writes adversarial NPZ batches and `results.json`.

Per seed, `S` is the certified successful-attack count, `N` the clean-correct attack-pool size and `D` the eligible pool size before correctness filtering. RA = (N − S)/D and ASR = S/N (when N > 0). Summary `RA` fields use percentages (0–100); batch-level `asr` fields use fractions (0–1). Multiply ratios by 100 when reporting percentages, and report five-seed mean ± SD.

## Code layout

- Root `*_run.py` / `run_official.py` and `run_*.sh`: experiment entries.
- `experiment_common.py`: shared dataset and victim-model loading.
- `method/runtime.py`: logits extraction, query accounting and completion helpers.
- `method/core/`: joint integer moves, completion, constraints and final checking.
- `method/widen_kernel_*.py`, `method/lcld_schedule_kernel.py`: frozen search schedules.
- `method/lcld_*`, `method/wids_mapping_*`: LCLD and mapping-specific components.
- `method/models/`: TabFPA victim-model adapters.
- `method/benchmark_adapter.py`: connection to the external benchmark and local asset paths.
- `targets/train_tree_targets.py`: training and loading the two additional tree targets.

The LCLD kernel is checked against `EXPECTED_KERNEL` in `method/lcld_config.py`. Its pinned hash reflects the released source file, including import paths. Generated outputs are ignored by Git.

## Acknowledgment and license status

Experiments use the datasets, target models and evaluation components of [TabularBench](https://github.com/serval-uni-lu/tabularbench), by Thibault Simonetto, Salah Ghamizi and Maxime Cordy. Please follow the upstream citation and asset-use instructions. TabFPA currently has no assigned top-level license.
