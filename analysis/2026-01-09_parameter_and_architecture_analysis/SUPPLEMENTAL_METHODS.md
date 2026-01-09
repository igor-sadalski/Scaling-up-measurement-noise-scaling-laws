# Supplemental Methods

## Geneformer

Geneformer is implemented as a BERT-based transformer model (BERTForMaskedLM) with 3 transformer layers, hidden dimension of 256, 4 attention heads, and intermediate size of 512. The model uses a maximum input sequence length of 512 tokens. The architecture uses ReLU activation, layer normalization epsilon of 1e-12, attention and hidden dropout of 0.02, and initializer range of 0.02.

The total parameter count is 13.5 million trainable parameters. The parameter count scales linearly with vocabulary size: the embedding layer and language model head each contribute `vocab_size × 256` parameters (~5-7.7M each), position embeddings contribute 131,072 parameters, and the three transformer layers contribute approximately 1.57M parameters total (~524,800 per layer from self-attention, feed-forward networks, and layer normalization).

Training is implemented using HuggingFace Trainer API. The model is optimized with AdamW using a maximum learning rate of 1e-3, linear decay schedule with 5,000 warmup steps, and weight decay of 0.001. Training uses batch size 64 per device with length-grouped batching enabled to minimize padding overhead for variable-length sequences. The maximum number of epochs is dynamically scaled as `max_epochs = max(1, int(10 × (10,000,000 / dataset_size)))` to ensure consistent computational budget across dataset sizes, ranging from 1 epoch for 10 million cells to 100,000 epochs for 100 cells. Early stopping is enabled with patience of 5 evaluation steps, monitoring validation loss every 1,000 steps. Evaluation and model saving occur every 1,000 steps, with automatic cleanup maintaining the 500 most recent checkpoints. The model uses masked language modeling loss with standard BERT-style random masking of 15% of tokens.

Data preprocessing involves ranking genes by their median expression values across the dataset and assigning token IDs based on rank. The token dictionary is created from all detected genes and stored as a pickle file. Each cell produces a variable-length sequence (up to 512 tokens) based on the number of expressed genes, with shorter sequences padded and longer sequences truncated. Tokenization uses parallel processing with up to 64 processes, partitioning data into chunks of 10-15 million cells, and storing tokenized data in HuggingFace Datasets format. Embedding extraction uses mean pooling over all gene tokens in the sequence from the final transformer layer, producing 256-dimensional cell embeddings processed in batches of 50 cells.

### Geneformer Training Parameters

| Parameter Category | Parameter | Value |
|-------------------|-----------|-------|
| **Architecture** | Model Type | BERT (BertForMaskedLM) |
| | Number of Layers | 3 |
| | Hidden Dimension | 256 |
| | Embedding Dimension | 256 |
| | Attention Heads | 4 |
| | Intermediate Size | 512 |
| | Max Input Sequence Length | 512 tokens |
| | Activation Function | ReLU |
| | Layer Normalization Epsilon | 1e-12 |
| | Attention Dropout | 0.02 |
| | Hidden Dropout | 0.02 |
| | Initializer Range | 0.02 |
| | Vocabulary Size | Variable (20,000-30,000 genes + 2 special tokens) |
| | Total Parameters | ~13.5M (13,480,151 for typical datasets) |
| **Optimization** | Optimizer | AdamW |
| | Maximum Learning Rate | 1e-3 |
| | Learning Rate Schedule | Linear decay |
| | Warmup Steps | 5,000 |
| | Weight Decay | 0.001 |
| **Training** | Batch Size (per device) | 64 |
| | Evaluation Batch Size (per device) | 100 |
| | Max Epochs | Dynamic: `max(1, int(10 × (10,000,000 / dataset_size)))` |
| | Epoch Range | 1 (10M cells) to 100,000 (100 cells) |
| | Early Stopping Patience | 5 evaluation steps |
| | Early Stopping Metric | Validation loss (minimize) |
| | Evaluation Strategy | Steps (every 1,000 steps) |
| | Save Strategy | Steps (every 1,000 steps) |
| | Save Total Limit | 500 checkpoints |
| | Load Best Model at End | True |
| | Group by Length | True |
| | Length Column Name | "length" |
| **Loss Function** | Loss Type | Masked Language Modeling (MLM) |
| | Masking Strategy | BERT-style random masking |
| | Masking Percentage | 15% of tokens |
| **Data Processing** | Tokenization Method | Median expression ranking |
| | Max Sequence Length | 512 tokens |
| | Sequence Padding | Shorter sequences padded |
| | Sequence Truncation | Longer sequences truncated |
| | Tokenization Processes | Up to 64 parallel processes |
| | Data Partition Size | 10-15 million cells per chunk |
| | Data Format | HuggingFace Datasets |
| | Train/Validation Split | 80% / 20% |
| **Embedding Extraction** | Embedding Style | Mean pooling |
| | Pooling Layer | Final transformer layer (-1) |
| | Embedding Dimension | 256 |
| | Inference Batch Size | 50 (default) |
| | Data Prefetching Processes | 30 |
| **Hardware** | Device | GPU (CUDA) |
| | Device Isolation | CUDA_VISIBLE_DEVICES |
| | Multi-GPU Support | Yes (via device isolation) |

**Figure: Training loss for the Geneformer trained for the developmental task on full 10 million cells and full quality dataset**

## SCVI

SCVI is implemented as a variational autoencoder using `scvi.model.SCVI` from scvi-tools, with PyTorch Lightning as the underlying framework. The architecture consists of encoder and decoder networks, each with 1 hidden layer of dimension 512, latent dimension of 16, dropout rate of 0.1, gene-specific dispersion parameters, zero-inflated negative binomial (ZINB) gene likelihood, and Gaussian latent distribution.

The parameter count is gene-dependent and scales with the number of input genes. For a dataset with `n_genes` genes, the total parameter count is approximately `524,288 + (n_genes × 2,048) + bias_terms`. This yields approximately 41 million parameters for datasets with ~20,000 genes, 51 million for ~25,000 genes, and 61 million for ~30,000 genes. The encoder contributes `(n_genes × 512) + (512 × 512) + (512 × 16 × 2)` parameters, while the decoder contributes `(16 × 512) + (512 × 512) + (512 × n_genes × 3)` parameters, where the factor of 3 accounts for mean, dispersion, and dropout outputs.

Training uses Adam optimizer with learning rate 1e-3, weight decay 1e-6, epsilon 0.01, and batch size 512. The maximum epochs are dynamically scaled as `max_epochs = max(1, int(1 × (10,000,000 / dataset_size)))`, ranging from 1 epoch for 10 million cells to 100,000 epochs for 100 cells. Early stopping monitors validation ELBO with patience of 5 epochs, minimum delta of 0.01, and checks every epoch. KL divergence warmup is applied over 1 epoch, gradually increasing the KL weight from 0.0 to 1.0 to prevent posterior collapse. Learning rate scheduling uses reduce-on-plateau with minimum learning rate 1e-6, monitoring validation ELBO. The loss function is the evidence lower bound (ELBO), combining reconstruction loss (negative log-likelihood under ZINB distribution) and KL divergence regularization. Data is split 80% training and 20% validation with shuffling.

SCVI operates directly on raw count data in AnnData (H5AD) format without preprocessing, as the ZINB likelihood model is designed for raw counts. The `setup_anndata()` function automatically handles data preparation. Embedding extraction uses the encoder to compute the mean of the latent distribution (posterior mean), producing 16-dimensional cell embeddings.

## PCA

PCA is implemented using sklearn TruncatedSVD. The method uses 16 components and computes a linear transformation matrix of size `n_genes × 16` where `n_genes` is the number of highly variable genes (750). Preprocessing involves total count normalization to 10,000 counts per cell using `scanpy.pp.normalize_total()`, followed by log(x + 1) transformation via `scanpy.pp.log1p()`. The top 750 highly variable genes are selected using Scanpy's `highly_variable_genes()` function, with the HVG mask pre-computed and stored as a pickle file. Only HVG-filtered data is used for PCA fitting. The TruncatedSVD model is fit via single-pass SVD decomposition on the normalized, HVG-filtered training data. The fitted model, including the HVG mask, is saved as a pickle file. Embedding extraction performs matrix multiplication of normalized, HVG-filtered data with the PCA transformation matrix, producing 16-dimensional embeddings. PCA is CPU-based and typically completes in seconds to minutes depending on dataset size.

## Random Projection

Random Projection is implemented using sklearn GaussianRandomProjection with 16 components and random state 42. The method has no trainable parameters; it uses a fixed random projection matrix of size `n_genes × 16` that can be regenerated from the random seed. The projection matrix is generated by sampling entries from a Gaussian distribution `N(0, 1/16)` with mean 0 and standard deviation 0.25. The fitting process uses only the first 5 cells of training data to establish matrix dimensions (input dimension equals number of genes, output dimension equals 16). The sparse matrix is converted to dense array format for fitting. The projection matrix is generated once based on the random seed and remains fixed, making the process deterministic and extremely fast (completing in milliseconds). The model is saved as a joblib pickle file. Random Projection operates directly on raw gene expression counts without normalization, maintaining sparse matrix representation until transformation. Embedding extraction performs matrix multiplication of input data (converted to dense array) with the random projection matrix, producing 16-dimensional embeddings. The method is CPU-based and provides theoretical guarantees via the Johnson-Lindenstrauss lemma for approximate preservation of pairwise distances.

## Engineering Scale and Computational Infrastructure

The implementation required substantial engineering effort to scale across multiple algorithms, datasets, and experimental conditions. Over 3,800 training runs were logged, systematically evaluating multiple single-cell datasets (Shendure, PBMC, Larry, MERFISH) across logarithmically spaced dataset sizes (e.g. 100 to 10 million cells, 10 points) and data qualities (10 quality points per dataset), with multiple random seeds (typically 3: 42, 1404, 2701) for robustness.

To that end we built a parallel execution infrastructure supporting up to 100 concurrent workers, with GPU management via `CUDA_VISIBLE_DEVICES` ensuring each job uses a dedicated GPU without interference. Memory management includes configurable limits per algorithm (e.g., 240GB for Geneformer, 33GB default). Data processing uses parallel tokenization with up to 64 processes, partitioning data into chunks of 10-15 million cells, and storing data in efficient formats (HuggingFace Datasets for tokenized data, H5AD for AnnData). All trained models and embeddings are saved to disk for reproducibility, with comprehensive logging to Weights & Biases tracking all training metrics, hyperparameters, and model configurations.

For Geneformer we use length-grouped batching to minimize padding overhead for variable-length sequences, which is critical for single-cell data where sequence lengths vary by orders of magnitude. Rather than using a classification token, Geneformer extracts cell-level embeddings via mean pooling over all gene tokens, allowing aggregation of information from all expressed genes regardless of sequence length. Geneformer tokenizes genes based on median expression values across the dataset, creating a ranking-based representation that is more robust to outliers and batch effects compared to absolute expression values, and naturally handles zero-inflated single-cell data. PCA uses TruncatedSVD instead of full eigendecomposition for computational efficiency, and operates on the top 750 highly variable genes rather than all genes to focus on informative variation while maintaining tractability. SCVI uses KL divergence warmup over 1 epoch to gradually introduce the regularization term and prevent posterior collapse, which is particularly important for single-cell data where the prior may not match the data distribution initially. Early stopping patience is set to 5 steps for Geneformer (monitoring every 1,000 steps) and 5 epochs for SCVI (monitoring every epoch), balancing training efficiency with convergence for transformer versus VAE architectures.
