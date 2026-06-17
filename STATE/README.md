# STATE Local Deployment

Local deployment of the [STATE](https://github.com/arc-institute/state) framework (Arc Institute) for learning self-supervised cell embeddings from spatial transcriptomics data, as part of the measurement noise scaling laws project.

## Structure

```
STATE/
├── train_state.py               # Main training script (data prep -> preprocess -> fit -> embed -> UMAP)
├── brainstorm_training.py       # Hyperparameter sweep: compares 4 training configs
├── se_config.yaml               # Custom Hydra config for local spatial data
├── state/                       # STATE package source (Arc Institute)
│   ├── src/state/
│   │   ├── emb/                 # SE (State Embedding) model
│   │   │   ├── nn/
│   │   │   │   ├── model.py            # StateEmbeddingModel (Lightning module)
│   │   │   │   ├── flash_transformer.py # Flash attention transformer encoder
│   │   │   │   └── loss.py             # Wasserstein, KL, MMD, TabularLoss
│   │   │   ├── data/loader.py          # H5adSentenceDataset, collator
│   │   │   ├── train/trainer.py        # Lightning trainer setup
│   │   │   └── inference.py            # Inference class for embedding generation
│   │   ├── tx/                  # TX (perturbation) model (not used locally)
│   │   │   └── models/
│   │   │       ├── base.py             # PerturbationModel base class
│   │   │       └── state_transition.py # StateTransitionPerturbationModel
│   │   ├── _cli/                # CLI handlers
│   │   │   ├── _emb/                   # state emb {preprocess,fit,transform,eval,query}
│   │   │   └── _tx/                    # state tx {train,predict,infer}
│   │   └── configs/             # Hydra configuration
│   │       ├── state-defaults.yaml     # SE default config
│   │       ├── config.yaml             # TX base config
│   │       └── model/*.yaml            # Model architecture configs (state, scgpt, cpa, scvi)
│   ├── README.md                # Full STATE documentation
│   ├── MODEL_LICENSE.md         # CC BY-NC-SA 4.0
│   └── MODEL_ACCEPTABLE_USE_POLICY.md
├── data/                        # Local h5ad data files
│   ├── raw_train_data.h5ad
│   ├── raw_val.h5ad
│   ├── raw_test_data.h5ad
│   └── test_embedded.h5ad      # Embeddings output
├── se_profile/                  # Preprocessing output (gene embeddings, mappings)
├── se_checkpoints/              # Trained model checkpoints
├── brainstorm/                  # Hyperparameter comparison outputs
│   ├── brainstorm_comparison.png
│   ├── baseline_lr1e3_ep10/
│   ├── low_lr1e4_ep50/
│   ├── vlow_lr5e5_ep100/
│   └── small_batch_lr1e4_ep50/
└── umap_test_embeddings.png     # UMAP visualization
```

## Usage

### Train SE model
```bash
python train_state.py
```
This runs the full pipeline: CSV manifest creation -> preprocessing -> training -> embedding -> UMAP visualization.

### Hyperparameter sweep
```bash
python brainstorm_training.py
```
Compares 4 configurations varying learning rate (1e-3 to 5e-5), epochs (10 to 100), and batch size (16 to 64).

### STATE CLI (direct)
```bash
state emb preprocess --config se_config.yaml
state emb fit --config se_config.yaml
state emb transform --config se_config.yaml
```

## Default Architecture

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| FFN hidden dim | 512 |
| Attention heads | 4 |
| Transformer layers | 3 |
| Output dim | 256 |
| Dropout | 0.02 |
| Context window (pad_length) | 512 |
| Loss | TabularLoss (Wasserstein + energy) |
| Optimizer | AdamW, max_lr=1e-3, weight_decay=0.001 |
| Gene embeddings | ESM-2 (default) or one-hot (spatial) |
