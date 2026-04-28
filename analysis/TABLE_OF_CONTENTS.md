# Table of Contents — `analysis/`

This directory holds the analysis pipeline for the *Scaling up measurement-noise scaling laws* project: hyperparameter fits, MI computation, model evaluation, plotting, and final figures. Files are dated `YYYY-MM-DD_HH-MM_<purpose>.{py,ipynb,sh,csv,json}`.

Conventions:
- `compute_*.py` → run the experiment / compute MI / collect results.
- `plot_*` / `plotting_*.ipynb` → make figures from the corresponding compute output.
- `prepare_train_lmi.ipynb` → end-to-end notebook (preprocess + train + LMI) for one (dataset, size, quality).
- `run_state_*.py` → orchestrators that launch many STATE training jobs through the project's `Experiments` API.

---

## 1. Top-level analysis files

### 1a. 2025 — original cell + noise scaling pipeline

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `2025-11-18_17-10_hyperparam_fits_cell_scaling.py` | Cell-number scaling fits | Fits power-law `I(N) = I_inf − (N/N0)^(−s)` via lmfit with bounds + uncertainty estimation | MI vs cell-count data per (method, metric, quality) | `fit_params (N0, s, I_inf)`, errors, residuals, uncertainty bands | Robust hyperparameter scan for cell scaling fits |
| `2025-11-18_17-20_hyperparm_fits_noise_scaling.py` | Noise scaling fits | Fits `I(u) = I_max − 0.5·log2(...)` to UMI-downsampled MI; multiprocessing across param combos | MI vs UMIs/cell data | `fitted_u_bar`, `I_max`, errors, uncertainty bands | Hyperparameter scan for noise scaling robustness |
| `2025-11-18_17-23_check_results.ipynb` | Error distribution analysis | Distributions of fitting errors for cell/noise scaling, residuals, uncertainty histograms | CSVs from fitting scripts | Histograms, median ratios | Validate fit quality, find outliers |
| `2025-11-18_17-40_select_cell_scaling_with_smalles_avg_uncertainty_error copy.ipynb` | Best-fit selection | Picks fits minimizing `avg_uncertainty_error` per (method, metric, quality) group | Intermediate cell-scaling-fits CSVs | Selected best-fit CSV | Choose highest-quality fits for final results |
| `2025-11-18_18_visualize_the_results_cell_scaling.py` | (Empty placeholder) | 0-byte file; never implemented | — | — | Intended cell-scaling visualization |
| `2025-11-18_18_visualize_the_results_noise_scaling.py` | Noise-scaling plots | Loads `noise_scaling.csv`, plots fitted curves with uncertainty bands per (method × metric × dataset) | `noise_scaling.csv` | Multi-panel figure with fits + CI bands | Visualize quality of noise-scaling fits |
| `2025-12-02_19-54_evaluate_mid_quality_models_on_full_data.py` | Mid-quality eval on full data | Uses `Experiments` orchestrator to score 0.1-UMI models on full-quality test sets across 4 datasets | Pre-trained checkpoints from S3 download | Embedding results, MI estimates | Test generalization when train/test quality differ |
| `2025-12-02_19_48_download_necessary_models.sh` | Bulk model download | `aws s3 cp` of pre-trained models, test data, utils for PBMC/larry/shendure/merfish at 0.1 UMI | S3 paths | Local `results/`, `preprocessed/`, `utils/` trees | Fetch infrastructure for mid-quality eval |
| `2025-12-05_19-23_verify_mi_of_models_trained_on_low_quality_data.ipynb` | Low-quality MI verification | Compares MI estimates of low-quality-trained models across methods & datasets | MI computation results | Plots, statistics on MI vs. quality | Confirm MI captures low-quality training signal |
| `2025-12-07_23-12_download_different_geneformer_models.sh` | Multi-accuracy Geneformer download | Pulls Geneformer models trained at 10 accuracy levels (1.0 → 0.004); copies test sets | S3 paths, accuracy list | `shendure/10000000/<accuracy>/results` tree | Enable Geneformer cross-quality eval |
| `2025-12-07_23-21_evaluate_all_shendure_geneformer_models_on_mid_quality_data.py` | Geneformer cross-quality eval | Evaluates Geneformer (any train quality) on quality 0.0252 via `Experiments` API | Pre-downloaded models, shendure 10M test | Embeddings, MI estimates | Quantify Geneformer robustness to quality shift |
| `2025-12-08_18-18_generate_new_scaling_curves_geneformer.ipynb` | Geneformer scaling curves | Loads Geneformer MI; refits noise scaling; merges CSVs | Geneformer MI CSVs | `scaling_curves_geneformer.csv` | Clean Geneformer-only scaling curves |
| `2025-12-08_22-10_download_data_to_retrain_scvi_models.sh` | SCVI training data download | Copies prior Geneformer outputs + S3 SCVI validation data for 3 quality levels | S3 paths | SCVI training data layout | Prepare data to retrain SCVI |
| `2025-12-08_22-15_reeval_scvis.py` | SCVI re-eval at all qualities | Trains SCVI on 10 qualities (max_workers=30); evaluates on all 10 qualities via `Experiments` | Preprocessed shendure 10M, 10 qualities | SCVI embeddings + MI per quality | Regenerate SCVI scaling curves |
| `2025-12-08_22-45_regenerate_scaling_noise_curve_scvi.py` | SCVI noise-curve targeted | Trains+evaluates SCVI on quality 0.0251984 and 0.5414548 for shendure | Preprocessed shendure | SCVI outputs + MI for targeted points | Refine SCVI noise-scaling curve |
| `2025-12-08_23-33_generate_new_scaling_curves_geneformer_and_scvi.ipynb` | Combined Geneformer+SCVI curves | Loads MI from both; refits noise + cell scaling; merges outputs | Geneformer/SCVI MI CSVs | `..._results.csv` (combined) | Unified scaling curves across two algorithms |
| `2025-12-08_23-33_generate_new_scaling_curves_geneformer_and_scvi_results.csv` | Combined-fit checkpoint | Single-row tabular dump of merged Geneformer+SCVI fits (shendure, 10M, 1.0, temporal MI) | — (output) | 1-row CSV | Checkpoint of combined analysis |
| `2026-01-08_17-29_training_curves.ipynb` | Training-curve viewer | Loads training data, plots loss / val curves across sizes & qualities | Training logs | Loss vs. epoch line plots | Monitor model learning dynamics |
| `2026-01-08_parameter_counts.ipynb` | Param-count tabulation | Counts trainable params per algorithm at embedding-dim 256 | Model architecture defs | Param-count table | Document model complexity |

### 1b. Undated legacy / figure notebooks

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `big_fig.ipynb` | Main multi-panel figure | 8-panel figure: scaling laws, collapse, parameter strips, examples, Caltech101 | `cell_scaling.csv`, `noise_scaling.csv`, Caltech101 data | Publication-ready PDF/PNG | Comprehensive results figure |
| `big_fig_2.ipynb` | Extended results figure | Sequences, TissueMNIST, parameter comparisons across modalities | Sequence MI, TissueMNIST CSVs, scaling results | Multi-dataset comparison plots | Extend universality across data types |
| `cell_number_scaling.ipynb` | Cell-scaling deep dive | Detailed `I(N)` exploration; collapse + exponent variation by metric/method | `collect_mi_results`, cell-scaling fit params | Exponent comparisons, collapse plots | Understand dataset-size dependence |
| `collapse.ipynb` | Universal scaling collapse | Plots `(I_inf − I)^(−1/s) vs N/N0` to test universality | Cell-scaling params, MI data | Log-log collapse plot | Validate theoretical scaling framework |
| `comparisons.ipynb` | Algorithm rankings | R² heatmaps + parameter ranks across Geneformer/PCA/SCVI | `fit_quality_results.csv`, scaling CSVs | Comparison tables/heatmaps | Benchmark algorithms |
| `errors.ipynb` | Error analysis | R², mean residuals, percent errors for fits | `collect_mi_results`, fit-param CSVs | R² heatmaps by method/metric | Validate model quality |
| `fit_quality_results.csv` | Fit-quality table | 191 curves × dataset, curve_identifier, R² across transcriptomics + Caltech101 + sequences + TissueMNIST | — (output) | Tabular goodness-of-fit | Document fit quality across domains |
| `merge_evaluation_results.py` | Eval-result merger | Loads evaluation CSV, merges with original MI; aligns datasets and signals | Evaluation CSV (12-02 output), original results CSV | Merged CSV with original + new LMI columns | Combine train/test eval metrics |
| `model_fit_quality.ipynb` | Fit-quality plots | Side-by-side R² heatmaps comparing noise vs cell scaling | `cell_scaling`, `noise_scaling` CSVs | R² heatmaps | Assess which scaling law fits better |
| `playground.ipynb` | Exploratory scratchpad | Ad-hoc analysis, rankings, parameter correlations | Various result CSVs | Exploratory plots/stats | Iterative hypothesis testing workspace |
| `projections.ipynb` | Future-scaling projection | Projects MI gain from 10× cells or 10× UMIs; heatmaps of % / bits gain | Fit params (N0, s, I_max, u_bar) | % and bits-gain heatmaps | Predict experimental gains |
| `refit.ipynb` | Refit with extra parameter | Tests 4-param cell-scaling model adding `I_0`; compares old vs new R² | `collect_mi_results`, fit CSVs | Diagnostics, R² deltas | Optimize model complexity |
| `shendure_only.ipynb` | Shendure-focused analysis | Scaling laws on largest (10M) dataset; temporal-MI-focused | shendure MI subset | Fitted curves, parameter comparisons | Stability of scaling at extreme size |
| `u90.ipynb` | u90 metric | Computes `u_0.9` (UMI level for 90% info capacity) per (method, metric); LaTeX-table output | `noise_scaling.csv` | u90 table with errors | Practical robustness threshold |

### 1c. 2026-04 — STATE / ESM / fine-tune / model-sizing pipeline

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `2026-04-13_14-19_s3_retriever_usage.ipynb` | S3Retriever API tutorial | Two workflows: load precomputed MI; retrieve `(embedding, signal)` pairs | — | Example queries, UMAPs colored by biology | Document unified retrieval API |
| `2026-04-13_18-33_state_se_pbmc_46k_prepare_train_lmi.ipynb` | STATE on PBMC 46k @ q=1.0 | `state emb preprocess` → train (256-dim, 3-layer transformer) → embed → LMI on `protein_counts` → compare to other algos | PBMC raw, ESM gene embeds | 256-dim embeddings, LMI ≈3.43 nats, ckpt, loss curves | Establish STATE baseline + validate pipeline |
| `2026-04-13_20-36_state_se_larry_46k_prepare_train_lmi.ipynb` | STATE on larry 46k @ q=1.0 | Same protocol as PBMC; LMI on `clone` signal | larry raw, ESM, preprocessed h5ad | 256-dim embeddings, LMI ≈0.93 nats, loss curves, UMAPs | Validate on lineage-tracing dataset |
| `2026-04-13_20-36_state_se_merfish_60k_prepare_train_lmi.ipynb` | STATE on MERFISH 60k @ q=1.0 | Same protocol; 99.2% gene/ESM overlap; LMI on `cur_idx`, `ng_idx` | merfish raw, ESM | 256-dim embeddings, LMI ≈1.26-1.33 nats, UMAPs | Validate on spatial transcriptomics |
| `2026-04-13_20-36_state_se_shendure_59k_prepare_train_lmi.ipynb` | STATE on shendure 59k @ q=1.0 | Same protocol; 47.9% ESM overlap; LMI on `author_day` | shendure raw, ESM | 256-dim embeddings, LMI ≈1.41 nats, UMAPs | Validate on large embryo atlas with sparse coverage |
| `2026-04-14_01-13_merge_esm_embeddings.ipynb` | Build merged ESM2 dictionary | Loads TranscriptFormer ESM2 (mouse + human, 2560-dim); maps gene names via mygene; aliases short names + Ensembl IDs; intersects with required genes | Raw dataset gene names + ESM2 mouse/human HDF5 | `merged_esm_embeddings.pt` (61,260 keys × 2560-dim), per-dataset coverage | Unified ESM2 vocab for STATE preprocessing across all datasets |
| `2026-04-14_11-28_state_untrained_baseline_pbmc_46k.ipynb` | Untrained STATE vs RP | Untrained STATE checkpoints at 512 / 2048 context; RP at 512 / 2048 / ESM-overlap / all 20.7k genes | PBMC test, untrained ckpts, ESM mask | LMI for 6 baselines (range 0.54–3.19 nats) | Ablate transformer's pretraining-independent contribution |
| `2026-04-14_14-24_run_state_merfish_whole.py` | Full-merfish STATE run | Runs STATE on the whole merfish dataset (largest size) across qualities | Preprocessed merfish | STATE outputs (ckpt, embeds, LMI) | Full-scale merfish baseline |
| `2026-04-14_15-25_collect_and_plot_mi_scaling.ipynb` | Master MI collector + plotter | Walks the data tree, aggregates `(dataset, size, quality, algo, signal, seed)` MI → CSV; log-log scaling curves with power-law fits | All `lmi_mutual_information.txt` files | `collect_mi_results.csv`, `collect_mi_results.png` | Empirical scaling curves across algos & datasets |
| `2026-04-14_16-03_preprocess_state_all.py` | STATE preprocessing (parallel) | `state emb preprocess` over all `(dataset, size, quality)`, 50 workers; builds gene-embed profile + valid-gene masks | Raw datasets, ESM, sampling configs | `state_data/` per (size, quality) | Prebuild STATE inputs so training jobs run in parallel |
| `2026-04-14_16-19_run_state_all_datasets.py` | Run STATE on 4 datasets | `Experiments.parallel_run` for PBMC/larry/merfish/shendure; 15k-step budget, early stop on val loss | Preprocessed STATE data | Ckpts, embeddings, LMI for 10×10×4 = 400 runs | Full STATE baseline matrix |
| `2026-04-14_16-22_run_state_merfish_size_and_quality.py` | merfish size + quality scaling | (10 sizes @ q=1.0) + (10 qualities @ max size); seed=42 | Preprocessed merfish | STATE results for 20 (size, quality) pairs | Fill in merfish size/quality scaling curves |
| `2026-04-14_16-24_state_se_pbmc_2154_q0475_prepare_train_lmi.ipynb` | STATE on PBMC 2154 @ q=0.475 | Train on small PBMC subset @ ~50% quality; LMI on `celltype.l3`, `protein_counts` | PBMC raw + downsampled counts | Embeddings, ckpt, LMI ≈2.51 nats | Probe low-sample mid-quality regime |
| `2026-04-14_16-47_clear_state_merfish_results.py` | Wipe merfish STATE outputs | Deletes `results/State/` for all (size, quality) to free disk | Path list | ~100+ dirs removed (~100+ GB) | Recovery before re-running with checkpoint resume |
| `2026-04-15_10-18_run_state_all_datasets.py` | Single-seed STATE on larry/merfish/shendure | One seed (42) per dataset; 2 jobs/GPU; PBMC skipped | Preprocessed STATE data | STATE results across 10×10 grids | Fast grid coverage, complement to multi-seed PBMC |
| `2026-04-15_14-43_compute_loss_scaling.py` | Test-loss collector | Trains supervised probes (kNN, logreg, ridge) on each `embedding.csv`; saves scalar metric `.txt`s; 8 workers | Embeddings + test signals | Per-experiment metric files | Downstream-task metric to supplement MI |
| `2026-04-15_14-56_plot_loss_scaling.ipynb` | Loss-scaling plots | Aggregates probe metrics → plots vs size / quality | Outputs of compute_loss_scaling | Multi-panel scaling plots | Visualize probe metric scaling |
| `2026-04-15_15-30_compute_linear_probe_scaling_curves.py` | Bulk LMI computation | `latentmi.lmi` per `(dataset, algo, size, quality, signal, seed)`; max_epochs=300, early stop; 8 workers | Embeddings + signals | `lmi_mutual_information.txt` per experiment | Full LMI grid across all algos |
| `2026-04-15_15-30_plot_linear_probe_scaling_curves.ipynb` | Linear-probe scaling plots | Loads LMI grid → log-log scaling curves with fits | LMI outputs | Scaling-curve PNGs | Visualize LMI scaling |
| `2026-04-16_08-50_run_state_missing.py` | STATE gap-filler | Scans for missing MI; reruns only those (size, quality) combos; order: larry, merfish, PBMC, shendure | Partial / missing MI results | Filled STATE outputs | Recover after crashes |
| `2026-04-16_14-43_compute_state_hparam_sweep_pbmc.py` | STATE hp sweep on PBMC | N trials × 10 sizes × 10 qualities; varies dropout/batch/lr/wd; 15k step budget | PBMC preprocessed, hp grid | `hp_tunning/sweep_results.csv`, per-trial yamls | Validate production STATE hyperparameters |
| `2026-04-16_14-43_plotting_state_hparam_sweep_pbmc.ipynb` | STATE hp-sweep plots | Visualizes MI vs each hyperparameter | hp-sweep CSVs | Heatmap / line plots | Identify robust hp settings |
| `2026-04-16_14-49_compute_state_all_datasets.py` | Standardized STATE full-grid run | 15k step budget, val every 1k, patience=5, LMI max_epochs=300; PBMC/larry/shendure | Preprocessed STATE data | Ckpts, embeddings, train/val/test loss, LMI | Uniform protocol for reproducible full grid |
| `2026-04-16_14-49_plotting_state_all_datasets.ipynb` | STATE full-grid plots | Loads STATE LMI across all datasets → scaling curves | LMI outputs | Multi-dataset scaling-curve PNGs | Visualize STATE scaling everywhere |
| `2026-04-17_11-32_plotting_state_loss_curves.ipynb` | STATE loss-curve plotter | Reads Lightning `metrics.csv` from each (size, quality); plots train/val log-log + best ckpt | `metrics.csv` files | Per-dataset loss-curve PNGs | Visualize STATE training dynamics |
| `2026-04-18_09-49_plotting_loss_curves_all_algos.ipynb` | All-algo loss curves | Reads Geneformer `trainer_state.json`, SCVI losses, STATE `metrics.csv`, PCA/RP loss logs; overlays by algo + size | Per-algo trainer logs | Multi-panel comparative loss-curve PNGs | Compare training stability across embedding methods |
| `2026-04-18_10-03_plotting_geneformer_best_step_vs_size.ipynb` | Geneformer best-step vs size | Identifies best val-loss step per (size, quality); plots step vs log(size) | `trainer_state.json` (eval history) | Scatter/line plots | Inform step-budget choices |
| `2026-04-18_10-42_plotting_geneformer_test_loss_vs_mi.ipynb` | Geneformer test-loss vs MI | Scatters `test_loss.txt` against `lmi_mutual_information.txt` | per-experiment loss/MI files | Scatter PNG | Validate test loss as MI proxy |
| `2026-04-20_10-49_compute_ksg_scaling_curves.py` | KSG MI on shendure SCVI | `latentmi.ksg.mi` (k=3) on raw 256-dim embeddings; 64 workers | Embeddings + signals | `ksg_mutual_information.txt` per experiment | Compare KSG vs LMI on high-dim embeddings |
| `2026-04-20_10-49_plotting_ksg_scaling_curves.ipynb` | KSG vs LMI scaling | Overlays log-log KSG and LMI scaling with power-law fits | KSG + LMI outputs | Dual-curve PNG | Validate scaling-law robustness across MI estimators |
| `2026-04-20_14-31_compute_state_model_sizing_pbmc.py` | STATE model-size sweep | Control 256/3 + 4 alt configs; constant ratios (`d_hid=2·emsize`, `output_dim=emsize`); 15k step budget | PBMC STATE data, sweep spec | `model_sizing/sweep_results.csv`, per-config yamls + LMI | Find optimal STATE capacity |
| `2026-04-20_14-31_plotting_state_model_sizing_pbmc.ipynb` | STATE model-size plots | MI vs `emsize`, lines per `nlayers`; size and quality scaling | Sweep yamls | Line plots, one panel per signal | Characterize STATE capacity scaling |
| `2026-04-20_15-26_plotting_geneformer_noise_scaling_pbmc.ipynb` | Geneformer noise scaling | MI vs quality (linear + log) with power-law fits | Geneformer LMI on PBMC | Noise-scaling PNGs | Quantify Geneformer noise sensitivity |
| `2026-04-21_10-55_compute_finetune_pretrained_geneformer_pbmc_noise_scaling.py` | Fine-tune gc104M Geneformer | Continues MLM (not classification) on each (size, quality); EPOCHS-fixed; extracts embeddings; LMI | gc104M ckpt, PBMC raw | Fine-tuned ckpts, embeddings, LMI | Evaluate 316M-param pretrained model on grid |
| `2026-04-21_13-47_compute_state_largest_shendure.py` | STATE on 3 largest shendure sizes | Skips complete combos; trims stale 1k-step ckpts; auto-resumes from `last.ckpt`; 15k max_steps | Crashed STATE shendure runs | Completed STATE results | Recover from disk-full crashes via ckpt resume |
| `2026-04-21_13-47_plotting_state_largest_shendure.ipynb` | Largest-shendure STATE plots | MI vs quality (log-log) with power-law fits, sizes (774k, 2.7M, 10M) | LMI outputs | Power-law noise-scaling PNGs | Test scaling laws at extreme scale |
| `2026-04-21_14-00_compute_finetune_pretrained_state_pbmc_noise_scaling.py` | Fine-tune SE-100M on PBMC grid | Downloads SE-100M, copies stripped ckpt as `last.ckpt`, 1-epoch fine-tune at lr=1e-5, computes MI | SE-100M ckpt + config, PBMC | `finetunning_state/` results, embeddings, LMI | Evaluate publicly available pretrained STATE |
| `2026-04-21_14-00_plotting_finetune_pretrained_state_pbmc_noise_scaling.ipynb` | SE-100M vs from-scratch | Overlays size/quality scaling for both; compares power-law exponents | LMI outputs | Dual-line PNGs | Quantify pretraining benefit |
| `2026-04-21_14-12_collect_metrics_and_loss_curves.ipynb` | Aggregate metrics across all experiments | Collects `train/val/test_loss.txt` per ckpt; aggregated trajectory CSV + summary plots | All ckpt log files | Aggregated loss CSV + summary PNGs | Unified view across ~2,000 experiments |
| `2026-04-22_14-31_compute_finetune_pretrained_geneformer_pbmc_noise_scaling.py` | Geneformer fine-tune (re-run) | Same as 2026-04-21_10-55 (likely with adjusted hp / extra seeds) | gc104M ckpt, PBMC | Fine-tuned ckpts, embeddings, LMI | Reproducibility / extra seed coverage |
| `2026-04-22_14-31_plotting_finetune_pretrained_geneformer_pbmc_noise_scaling.ipynb` | Fine-tune Geneformer plots | Plots fine-tuned Geneformer size/quality scaling vs from-scratch | LMI outputs | Scaling PNGs | Assess pretraining benefit for Geneformer |
| `2026-04-22_15-49_compute_geneformer_model_sizing_pbmc.py` | Geneformer model-size sweep | Control 256/3 + 6 alt configs (3 smaller intentionally tiny, 3 larger); constant head dim 64, max_input 512; 15k steps | PBMC Geneformer data | `model_sizing_geneformer/sweep_results.csv`, per-config yamls + MI | Find Geneformer capacity scaling vs STATE |
| `2026-04-22_15-49_plotting_geneformer_model_sizing_pbmc.ipynb` | Geneformer model-size plots | MI vs `num_embed_dim`, lines per `num_layers` | Sweep yamls | Line plots per signal | Compare Geneformer vs STATE capacity scaling |
| `2026-04-24_11-49_compute_shendure_geneformer_checkpoint_mi.py` | Per-checkpoint MI for shendure Geneformer | 20 evenly-spaced checkpoints per (size, quality); LMI on each in sequence; 32 workers across (size, quality) | Geneformer `checkpoint-*/` dirs, test signal | Per-ckpt MI files; CSV + JSON logs | Study training-trajectory MI evolution |
| `2026-04-24_11-49_compute_shendure_geneformer_checkpoint_mi.csv` | Per-ckpt MI table | Rows: size, quality, checkpoint, mi, status, error | (Output) | CSV from compute script | Track per-ckpt MI with run status |
| `2026-04-24_11-49_compute_shendure_geneformer_checkpoint_mi.json` | Per-ckpt MI structured log | JSON array of jobs (`size, quality, results=[{checkpoint, mi, status, error, raw_mi_map}]`) | (Output) | JSON log | Fault-recovery + result inspection |
| `2026-04-24_11-49_plotting_shendure_geneformer_checkpoint_mi.ipynb` | Geneformer-ckpt MI plots | MI vs ckpt number per size/quality | Per-ckpt LMI files | Line plots, panels per size/quality | Visualize MI convergence + over-/under-fitting |
| `2026-04-24_19-00_compute_update_state_larry_merfish.py` | Append STATE rows to master MI CSV | Walks larry + merfish STATE results, appends rows to `collect_mi_results.csv` (schema: dataset, size, quality, algo, signal, seed, mi_value, umis_per_cell); regenerates PNG | STATE MI files, existing master CSV | Updated `collect_mi_results.csv` + `.png` | Fill missing STATE rows + regenerate master plots |

---

## 2. `2026-01-09_parameter_and_architecture_analysis/`

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `SUPPLEMENTAL_METHODS.tex` | Supplemental methods (LaTeX source) | Documents Geneformer architecture, hyperparameters, training (dynamic-epoch scaling, early stopping, MLM); param counts (1.9M–13.5M); justifies reduced-param design vs STATE/Tahoe/scGPT | Theory + architecture docs | Source for PDF | Reproducibility + comparison with large-scale models |
| `SUPPLEMENTAL_METHODS.pdf` | Compiled supplementary methods | PDF rendering of `SUPPLEMENTAL_METHODS.tex` | LaTeX source + figure | Distributable PDF | Publication |
| `loss_train_eval_geneformer_developmental_10milion_quality1.png` | Geneformer 10M @ q=1.0 loss curves | Plots training + validation loss | Training logs | PNG figure | Show stable convergence (no overfit) on 10M cells |
| `other_models/` (subdir) | Placeholder folders for other algos | 6 empty subdirs: UCE, geneformer, scGPT, state, tahoe-x1, transcriptformer | — | Directory structure | Infrastructure for future cross-algo comparison |
| LaTeX build artifacts (`.aux`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz`) | pdflatex byproducts | Generated during PDF compile | TeX source | Index / sync files | Incremental builds + editor sync |

---

## 3. `2026-04-08_veryfing_data_correctness/`

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `collect_mi_results_from_disk.csv` | Aggregated MI for non-STATE algos | Scans disk for all MI files (4000 expected); reads scalar values; groups by (dataset, size, quality, algo, signal, seed); 100% complete | MI files on disk | CSV: dataset, size, quality, algo, signal, seed, mi_value | Single unified table for downstream analysis |
| `collect_mi_results_from_disk.ipynb` | Data-validation notebook | Parallel scan (128 workers); existence + value checks; per-algo per-dataset completeness pivots; validates embeddings + models | Local data tree | 3 CSVs + completeness pivots | Audit experiment-data integrity before analysis |
| `collect_state_results_from_disk.csv` | STATE-specific MI table | Same as above for STATE signals (`celltype.l3` for PBMC, `cur_idx` for MERFISH); 2100 of 2400 found (300 missing on merfish/Geneformer) | STATE MI files | CSV with same schema | Track STATE separately from baseline algos |
| `local_data_completeness.ipynb` | Completeness summary | Compares local training configs vs expected | Local data dirs | Summary report | QA before running missing MI |
| `mi_progress.ipynb` | MI-job progress monitor | Tracks MI completion during/after distributed runs | Job logs + result files | Progress table/plots | Real-time visibility into long compute |
| `missing_mi_on_disk.csv` | Gap report (baseline) | Empty (header-only) — all 4000 baseline configs complete | Filesystem scan | Empty CSV | Reproducibility — would list re-run targets if any failed |
| `missing_state_on_disk.csv` | Gap report (STATE) | 300 rows of missing STATE configs (all `merfish`, `Geneformer` branch) | Filesystem scan | CSV: 300 rows | Input to `run_missing_mi.py` for targeted re-run |
| `plot_mi_scaling.ipynb` | MI-scaling plots | Power-law scaling curves across sizes/qualities + algo comparisons | `collect_mi_results_from_disk.csv` | Scaling-curve PNGs | Visualize size + noise scaling |
| `run_missing_mi.py` | Targeted re-runner | Reads `missing_*_on_disk.csv`; groups by (dataset, size, quality, algo, seed); runs `Experiments.single_job()` in `ProcessPoolExecutor`; retrain/reembed/recompute flags | Gap CSV, `NOISE_SCALING_DATA_DIR` | Job-status dict + new MI files on disk | Recompute failed/missing MI jobs in parallel |

---

## 4. `2026-04-14_initial_results_state_with_esm/`

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `generate_figures.py` | Generates the 14 report figures | Hardcoded MI tables (PCA/RP/SCVI/Geneformer/State × PBMC/larry/MERFISH/shendure); creates bars + error bars, LMI comparisons, loss curves, UMAPs, loss heatmaps | MI dict embedded in script | `figures/*.png` (14 files) | Visual results for `report.pdf` |
| `report.pdf` | STATE-with-ESM2 initial-results report | PDF rendering; key finding: STATE underperforms (LMI <0.5 on PBMC/larry/shendure vs 1–4 for baselines), OK on MERFISH (99.2% coverage); discusses embedding collapse + ESM coverage gaps (44.9–99.2%) | LaTeX source + figures | Distributable PDF | Communicate preliminary STATE-ESM findings |
| `report.tex` | Report LaTeX source | Describes STATE SE architecture, ESM2 prep (93k gene keys, per-dataset coverage), 4-dataset results + comparison tables; hypothesizes embedding collapse from high-dim (2560-d) ESM input | Theory + result tables | `report.pdf` | Methodology documentation |
| `figures/` (subdir) | 14 PNG figures for the report | Includes `PBMC_celltype`, `PBMC_protein`, `esm_coverage`, `larry_clone`, `larry_lmi`, `larry_loss_curves`, `larry_umap`, `merfish_cur_idx`, `merfish_ng_idx`, `pbmc_lmi`, `pbmc_loss_curves`, `pbmc_umap`, `shendure_author_day`, `summary_comparison` | MI values, training logs, embeddings | PNG plots | Illustrate STATE vs baselines, loss curves, UMAPs |

---

## 5. `final_results/`

CSV/PNG pairs share a name: the `.csv` is the data, the `.png` is the matching figure.

| File | Short description | What it does (how) | Inputs | Outputs | Motivation |
|------|-------------------|--------------------|--------|---------|------------|
| `cell_scaling.csv` | Power-law fits for `MI vs N` | Fits `I(N) = (I_inf − I_0) · exp(−N0/N)^s` per (dataset, algo, signal, quality); reports params + errors + residuals (11 cols) | `collect_mi_results.csv` | ~4000 rows | Predict MI gains from larger data |
| `collect_mi_results.csv` | Master MI table | Joins MI values with `umis_per_cell` (UMI sampling depth) | Raw MI files + metadata | 4000+ rows | Central table for downstream analysis |
| `collect_mi_results.png` | Master MI heatmap | MI indexed by (size × quality × algo) | `collect_mi_results.csv` | Single heatmap | Quick scan of performance landscape |
| `finetune_se_100m_state_pbmc_noise_scaling.csv` | Fine-tuned SE-100M noise scaling | MI vs quality on PBMC for the 100M-param fine-tuned STATE | Training/embed results | Rows: dataset, method, signal, size, quality, mi | Benchmark large STATE variant noise robustness |
| `finetune_se_100m_state_pbmc_noise_scaling.png` | Fine-tuned STATE curve | MI vs quality for fine-tuned + baselines | matching `.csv` | Curve PNG | Compare large vs small STATE noise sensitivity |
| `geneformer_loss_curves_sml.csv` | Geneformer training curves (small datasets) | train/val loss vs step for PBMC/larry/MERFISH/shendure at smallest sizes; flags best ckpt | Training logs | Long table: dataset, size, quality, size_rank, curve, step, loss, is_best | Training dynamics in data-limited regime |
| `geneformer_loss_curves_sml.png` | Geneformer small-N loss curves | Train/val curves overlaid across datasets | matching `.csv` | Multi-line PNG | Show convergence on smallest datasets |
| `geneformer_loss_scaling.csv` | Final val loss vs size | 10 sizes × 10 qualities = 400 rows | Training logs | dataset, size, quality, final_val_loss | Quantify overfitting reduction with more data |
| `geneformer_loss_scaling.png` | Loss scaling-law curve | Final loss vs size on log-log; power-law fit | matching `.csv` | PNG | Estimate loss exponent |
| `geneformer_model_size_sweep.csv` | Geneformer arch sweep | Varies `embedding_dim`, `inter_size`, heads, layers on PBMC 100k; tracks `trainable_params_M` and `mi_protein_counts` | Training results | Wide table | Find optimal architecture |
| `geneformer_model_size_sweep.png` | Geneformer model-size scaling | MI vs `trainable_params_M` | matching `.csv` | Scatter ± power-law fit | Capacity-scaling for Geneformer |
| `hyperpam_sweep.csv` | STATE hp tuning | Grid over (max_lr, batch, dropout, wd) on PBMC 100k; records trial id, params, final MI | Hp trials | Wide table | Find best STATE hyperparameters |
| `hyperpam_sweep.png` | Hp-sweep heatmap | MI heatmap indexed by hp axes; highlights best | matching `.csv` | Heatmap PNG | Visualize hp sensitivity |
| `ksg_vs_quality_shendure_SCVI.csv` | KSG vs LMI on shendure SCVI | KSG (k-NN) vs LMI estimates by quality + size | SCVI embeddings, shendure data | Table: probe, dataset, algo, quality, size, mi_bits | Validate scaling laws aren't an LMI artifact |
| `ksg_vs_quality_shendure_SCVI.png` | KSG-vs-LMI plot | Overlays both estimators across quality | matching `.csv` | Dual-curve PNG | Robustness check across MI estimators |
| `linear_probe_scaling.csv` | Linear-probe R² vs (size, quality) | Ridge regression on embeddings predicting label; reports `mean_r2` | Embeddings + labels | Long table | Alt downstream metric vs MI |
| `linear_probe_scaling.png` | Linear-probe scaling curves | R² vs quality / size per algo | matching `.csv` | Multi-curve PNG | Compare probe-task to MI scaling |
| `noise_scaling.csv` | Power-law fits for `MI vs u` | Fits `I(u) = I_max − (u/u_bar)^β` per (dataset, algo, metric, size); reports params + errors (9 cols) | `collect_mi_results.csv` | ~4000 rows | Predict MI vs measurement quality |
| `shendure_geneformer_checkpoint_mi.csv` | Per-ckpt MI on shendure Geneformer | MI at each training ckpt (1k, 2k, …) per quality | Training ckpts | Long table: dataset, algo, signal, seed, size, quality, step, mi | Convergence speed + early-stopping decisions |
| `shendure_geneformer_checkpoint_mi.png` | Per-ckpt MI curves | MI vs step per quality; saturation point | matching `.csv` | Multi-line step plot | Visualize learning trajectory |
| `state_model_size_sweep.csv` | STATE model-size sweep on PBMC | STATE arch ablation; `trainable_params_M`, `mi_protein_counts` | STATE training | Table | STATE capacity scaling |
| `state_model_size_sweep.png` | STATE model-size scaling | MI vs `trainable_params_M` | matching `.csv` | Scatter ± fit | Compare STATE capacity vs Geneformer |
