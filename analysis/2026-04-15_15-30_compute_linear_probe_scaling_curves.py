"""
Compute supervised ML metrics on embeddings and save per-experiment result files.

For each (dataset, algorithm, size, quality) combination, trains a simple model
on the embeddings and saves individual metric .txt files alongside embeddings.csv.

Shendure (classification): knn_accuracy.txt, knn_macro_f1.txt, logreg_accuracy.txt, logreg_macro_f1.txt
PBMC (regression):         ridge_mean_r2.txt, ridge_mean_mse.txt, ridge_mean_mae.txt
merfish (regression):      ridge_mean_r2.txt, ridge_mean_mse.txt, ridge_mean_mae.txt
larry (regression):        ridge_mean_r2.txt, ridge_mean_mse.txt, ridge_mean_mae.txt
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'

import sys
from pathlib import Path

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


_log_fh = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)

import scaling_laws  # noqa: F401 — activates timestamped print
print(f"Logging to {LOG_PATH}")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import anndata as ad
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from scaling_laws.paths import DATA_DIR as DATA_ROOT
SEED = 42
N_WORKERS = 8
MAX_SAMPLES = 100_000

REGRESSION_METRICS = ['ridge_mean_r2', 'ridge_mean_mse', 'ridge_mean_mae']
CLASSIFICATION_METRICS = ['knn_accuracy', 'knn_macro_f1', 'logreg_accuracy', 'logreg_macro_f1']

ALGOS = ['Geneformer', 'PCA', 'RandomProjection', 'SCVI', 'State']

EXPECTED = {
    'PBMC': {
        'sizes': [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        'qualities': [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422, 0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
        'signal': 'protein_counts',
        'task': 'regression',
    },
    'larry': {
        'sizes': [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        'qualities': [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311, 0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
        'signal': 'clone',
        'task': 'regression',
    },
    'merfish': {
        'sizes': [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        'qualities': [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        'signal': 'ng_idx',
        'task': 'regression',
    },
    'shendure': {
        'sizes': [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
        'qualities': [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384, 0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        'signal': 'author_day',
        'task': 'classification',
    },
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def embeddings_dir(dataset, size, quality, algorithm):
    return DATA_ROOT / dataset / str(size) / str(quality) / 'results' / algorithm / 'model'


def embeddings_path(dataset, size, quality, algorithm):
    return embeddings_dir(dataset, size, quality, algorithm) / 'embeddings.csv'


def signal_path(dataset, quality, signal, algorithm):
    q = str(quality)
    if algorithm == 'Geneformer':
        return DATA_ROOT / dataset / 'test' / q / 'signals' / f'Y_{signal}_{q}_geneformer.csv'
    return DATA_ROOT / dataset / 'test' / q / 'signals' / f'Y_{signal}_{q}.csv'


# ---------------------------------------------------------------------------
# Subsampling helper
# ---------------------------------------------------------------------------
def _subsample(*arrays, max_n=MAX_SAMPLES, seed=SEED):
    """Subsample arrays to at most max_n rows (consistent across arrays)."""
    n = arrays[0].shape[0]
    if n <= max_n:
        return arrays if len(arrays) > 1 else arrays[0]
    idx = np.random.RandomState(seed).choice(n, max_n, replace=False)
    out = tuple(a[idx] for a in arrays)
    return out if len(out) > 1 else out[0]


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------
def evaluate_shendure(emb_path, sig_path):
    X = pd.read_csv(emb_path).values.astype(np.float64)
    y_raw = pd.read_csv(sig_path).iloc[:, 0].values
    y = LabelEncoder().fit_transform(y_raw)
    X, y = _subsample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    res = {
        'knn_accuracy': accuracy_score(y_test, y_pred),
        'knn_macro_f1': f1_score(y_test, y_pred, average='macro'),
    }

    lr = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    res['logreg_accuracy'] = accuracy_score(y_test, y_pred)
    res['logreg_macro_f1'] = f1_score(y_test, y_pred, average='macro')
    return res


def evaluate_pbmc(emb_path, sig_path):
    X = pd.read_csv(emb_path).values.astype(np.float64)
    Y = pd.read_csv(sig_path).values.astype(np.float64)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)

    r2s = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    mses = [mean_squared_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    maes = [mean_absolute_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    return {'ridge_mean_r2': np.mean(r2s), 'ridge_mean_mse': np.mean(mses), 'ridge_mean_mae': np.mean(maes)}


def evaluate_merfish(emb_path, sig_path, quality):
    embeddings_df = pd.read_csv(emb_path)
    signal_df = pd.read_csv(sig_path, dtype=str)
    cur_idx, ng_idx = signal_df.iloc[:, 0].values, signal_df.iloc[:, 1].values

    assert len(cur_idx) == len(embeddings_df)
    assert set(ng_idx).issubset(set(cur_idx))

    embeddings_df.index = cur_idx
    Y = embeddings_df.loc[ng_idx].values

    preprocessed_path = DATA_ROOT / 'merfish' / 'test' / str(quality) / 'preprocessed' / 'preprocessed.h5ad'
    adata = ad.read_h5ad(preprocessed_path, backed='r')
    test_indices = adata.uns['test_indices'].astype(int)
    X = embeddings_df.values
    test_indices = test_indices[test_indices < X.shape[0]]
    X = X[test_indices].astype(np.float64)
    Y = Y[test_indices].astype(np.float64)
    assert X.shape == Y.shape

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)

    r2s = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    mses = [mean_squared_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    maes = [mean_absolute_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    return {'ridge_mean_r2': np.mean(r2s), 'ridge_mean_mse': np.mean(mses), 'ridge_mean_mae': np.mean(maes)}


def evaluate_larry(emb_path, sig_path):
    embeddings_df = pd.read_csv(emb_path)
    signal_df = pd.read_csv(sig_path, dtype={0: str, 1: str, 2: float})
    signal_df.columns = ['index', 'clone', 'time']

    early = signal_df[signal_df['time'].isin([2, 4])]
    late = signal_df[signal_df['time'].isin([6])]
    common_clones = (
        set(late['clone'].values)
        .intersection(set(early['clone'].values))
        .intersection(set(signal_df['index'].values))
    )
    if len(common_clones) == 0:
        return {'ridge_mean_r2': np.nan, 'ridge_mean_mse': np.nan, 'ridge_mean_mae': np.nan}

    early_common = early[early['clone'].isin(common_clones)].groupby('clone').sample(n=1, random_state=SEED)
    late_common = late[late['clone'].isin(common_clones)].groupby('clone').sample(n=1, random_state=SEED)

    embeddings_df.index = signal_df['index'].values
    X = embeddings_df.loc[early_common['index'].tolist()].values.astype(np.float64)
    Y = embeddings_df.loc[late_common['index'].tolist()].values.astype(np.float64)
    assert X.shape == Y.shape

    if X.shape[0] < 10:
        return {'ridge_mean_r2': np.nan, 'ridge_mean_mse': np.nan, 'ridge_mean_mae': np.nan}

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)

    r2s = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    mses = [mean_squared_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    maes = [mean_absolute_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
    return {'ridge_mean_r2': np.mean(r2s), 'ridge_mean_mse': np.mean(mses), 'ridge_mean_mae': np.mean(maes)}


# ---------------------------------------------------------------------------
# Save metric files
# ---------------------------------------------------------------------------
def save_metrics(out_dir, metrics):
    """Save each metric as a separate .txt file in out_dir."""
    out_dir = Path(out_dir)
    for name, value in metrics.items():
        if not np.isnan(value):
            (out_dir / f'{name}.txt').write_text(str(value))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_single_eval(args):
    dataset, algo, size, quality = args
    cfg = EXPECTED[dataset]
    emb_p = embeddings_path(dataset, size, quality, algo)
    sig_p = signal_path(dataset, quality, cfg['signal'], algo)

    if not emb_p.exists() or not sig_p.exists():
        return None

    out_dir = embeddings_dir(dataset, size, quality, algo)

    try:
        if dataset == 'shendure':
            metrics = evaluate_shendure(emb_p, sig_p)
        elif dataset == 'PBMC':
            metrics = evaluate_pbmc(emb_p, sig_p)
        elif dataset == 'merfish':
            metrics = evaluate_merfish(emb_p, sig_p, quality)
        elif dataset == 'larry':
            metrics = evaluate_larry(emb_p, sig_p)
        else:
            return None
    except Exception as e:
        return {'dataset': dataset, 'algorithm': algo, 'size': size,
                'quality': quality, '_error': str(e)}

    save_metrics(out_dir, metrics)

    return {
        'dataset': dataset, 'algorithm': algo, 'size': size,
        'quality': quality, 'signal': cfg['signal'], 'task': cfg['task'],
        **metrics,
    }


# ---------------------------------------------------------------------------
# Identify missing runs
# ---------------------------------------------------------------------------
def find_missing_jobs():
    """Return only jobs where at least one metric file is missing but embeddings exist."""
    all_jobs = []
    already_done = 0
    no_embeddings = 0

    for ds, cfg in EXPECTED.items():
        metric_names = CLASSIFICATION_METRICS if cfg['task'] == 'classification' else REGRESSION_METRICS
        for algo, size, quality in product(ALGOS, cfg['sizes'], cfg['qualities']):
            model_dir = embeddings_dir(ds, size, quality, algo)
            emb_p = model_dir / 'embeddings.csv'
            sig_p = signal_path(ds, quality, cfg['signal'], algo)

            # Skip if embeddings or signal don't exist (can't run anyway)
            if not emb_p.exists() or not sig_p.exists():
                no_embeddings += 1
                continue

            # Check if all metric files already exist
            if all((model_dir / f'{m}.txt').exists() for m in metric_names):
                already_done += 1
                continue

            all_jobs.append((ds, algo, size, quality))

    return all_jobs, already_done, no_embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    total_expected = sum(
        len(cfg['sizes']) * len(cfg['qualities']) * len(ALGOS)
        for cfg in EXPECTED.values()
    )

    jobs, already_done, no_embeddings = find_missing_jobs()

    print(f'Total expected:    {total_expected}')
    print(f'Already done:      {already_done}')
    print(f'No embeddings/sig: {no_embeddings}')
    print(f'Missing (to run):  {len(jobs)}')
    print(f'Workers:           {N_WORKERS}')

    if not jobs:
        print('\nNothing to do — all metrics are up to date.')
    else:
        results = []
        errors = []

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(run_single_eval, job): job for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Evaluating'):
                result = future.result()
                if result is None:
                    continue
                if '_error' in result:
                    errors.append(result)
                else:
                    results.append(result)

        df_results = pd.DataFrame(results)
        print(f'\nComputed {len(df_results)} results, {len(errors)} errors')
        if errors:
            print('First errors:')
            for e in errors[:5]:
                print(f"  {e['dataset']}/{e['algorithm']}/{e['size']}/{e['quality']}: {e['_error'][:120]}")

        # Completeness table
        if len(df_results) > 0:
            pivot = df_results.groupby(['dataset', 'algorithm']).size().unstack(fill_value=0)
            print('\nNew results per dataset x algorithm:')
            print(pivot)
