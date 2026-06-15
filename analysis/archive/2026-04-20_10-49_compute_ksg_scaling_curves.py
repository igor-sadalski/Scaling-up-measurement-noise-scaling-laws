"""
Compute KSG mutual information between embeddings and signal across all
(dataset, algorithm, size, quality) combinations, and save one
`ksg_mutual_information.txt` per experiment.

Mirrors the structure of `2026-04-15_15-30_compute_linear_probe_scaling_curves.py`
but uses `latentmi.ksg.mi` (KSG / Kraskov-Stoegbauer-Grassberger estimator)
instead of supervised probes.

Notes:
- `embeddings.csv` already contains TEST-SET embeddings (e.g. PBMC size=10000
  has 27 503 rows = test split), so KSG runs on the small test set, not the
  training size in the path.
- KSG returns ~0 MI when joint dimensionality is high (>~10) because of the
  curse of dimensionality. We do NOT reduce dimensions here — the raw KSG
  estimate on full embeddings/signal is what we want.
- `ksg.mi` returns a per-point array; we take `np.nanmean` for the scalar.

Output path:
    data/<dataset>/<size>/<quality>/results/<algorithm>/model/MI/ksg/<signal_stem>/ksg_mutual_information.txt
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path

# ── Auto-log: tee stdout/stderr to .log file next to this script ─────────
SCRIPT_PATH = Path(__file__).resolve()
LOG_PATH = SCRIPT_PATH.with_suffix(".log")


class Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


_log_f = open(LOG_PATH, "w")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
print(f"Logging to {LOG_PATH}")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from latentmi import ksg

from scaling_laws.paths import DATA_DIR as DATA_ROOT
SEED = 42
N_WORKERS = 64         # 96 CPUs available; leave headroom for I/O + tree builds
KSG_K = 3              # standard k for KSG estimator

ALGOS = ['SCVI']

# Only run KSG on these datasets. merfish/larry loaders remain defined below
# but are skipped by find_missing_jobs and total_expected.
ACTIVE_DATASETS = ('shendure',)

EXPECTED = {
    'PBMC': {
        'sizes': [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        'qualities': [0.0012346, 0.0025982, 0.0054682, 0.0115083, 0.02422,
                      0.050973, 0.1072766, 0.225772, 0.4751547, 1.0],
        'signal': 'protein_counts',
    },
    'larry': {
        'sizes': [100, 215, 464, 1000, 2154, 4641, 10000, 21544, 46415, 100000],
        'qualities': [0.003876, 0.0071835, 0.0133136, 0.0246748, 0.0457311,
                      0.0847557, 0.1570821, 0.2911284, 0.5395631, 1.0],
        'signal': 'clone',
    },
    'merfish': {
        'sizes': [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000],
        'qualities': [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267,
                      0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0],
        'signal': 'ng_idx',
    },
    'shendure': {
        'sizes': [100, 359, 1291, 4641, 16681, 59948, 215443, 774263, 2782559, 10000000],
        'qualities': [0.004, 0.0073875, 0.0136438, 0.0251984, 0.0465384,
                      0.0859506, 0.1587401, 0.2931733, 0.5414548, 1.0],
        'signal': 'author_day',
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


def ksg_output_path(dataset, size, quality, algorithm, signal_stem):
    return (embeddings_dir(dataset, size, quality, algorithm)
            / 'MI' / 'ksg' / signal_stem / 'ksg_mutual_information.txt')


# ---------------------------------------------------------------------------
# Per-dataset (X, Y) construction — mirrors BaseAlgorithm.mutual_information.
# No subsampling: KSG runs on the full aligned (X, Y) for each experiment.
# ---------------------------------------------------------------------------
def load_xy_pbmc(emb_path, sig_path):
    X = pd.read_csv(emb_path).values.astype(np.float64)
    Y = pd.read_csv(sig_path).values.astype(np.float64)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def load_xy_shendure(emb_path, sig_path):
    X = pd.read_csv(emb_path).values.astype(np.float64)
    sig = pd.read_csv(sig_path).iloc[:, 0].values
    Y = pd.get_dummies(sig).values.astype(np.float64)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def load_xy_merfish(emb_path, sig_path, quality):
    embeddings_df = pd.read_csv(emb_path)
    signal_df = pd.read_csv(sig_path, dtype=str)
    cur_idx, ng_idx = signal_df.iloc[:, 0].values, signal_df.iloc[:, 1].values

    assert len(cur_idx) == len(embeddings_df)
    assert set(ng_idx).issubset(set(cur_idx))

    embeddings_df.index = cur_idx
    Y = embeddings_df.loc[ng_idx].values

    preprocessed_path = (DATA_ROOT / 'merfish' / 'test' / str(quality)
                         / 'preprocessed' / 'preprocessed.h5ad')
    adata = ad.read_h5ad(preprocessed_path, backed='r')
    test_indices = adata.uns['test_indices'].astype(int)

    X = embeddings_df.values
    test_indices = test_indices[test_indices < X.shape[0]]
    X = X[test_indices].astype(np.float64)
    Y = Y[test_indices].astype(np.float64)
    assert X.shape == Y.shape
    return X, Y


def load_xy_larry(emb_path, sig_path):
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
        return None, None

    early_common = early[early['clone'].isin(common_clones)].groupby('clone').sample(n=1, random_state=SEED)
    late_common = late[late['clone'].isin(common_clones)].groupby('clone').sample(n=1, random_state=SEED)

    embeddings_df.index = signal_df['index'].values
    X = embeddings_df.loc[early_common['index'].tolist()].values.astype(np.float64)
    Y = embeddings_df.loc[late_common['index'].tolist()].values.astype(np.float64)
    assert X.shape == Y.shape
    if X.shape[0] < KSG_K + 2:
        return None, None
    return X, Y


def load_xy(dataset, emb_path, sig_path, quality):
    if dataset == 'PBMC':
        return load_xy_pbmc(emb_path, sig_path)
    if dataset == 'shendure':
        return load_xy_shendure(emb_path, sig_path)
    if dataset == 'merfish':
        return load_xy_merfish(emb_path, sig_path, quality)
    if dataset == 'larry':
        return load_xy_larry(emb_path, sig_path)
    raise ValueError(f'Unknown dataset {dataset}')


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_single_ksg(args):
    dataset, algo, size, quality = args
    cfg = EXPECTED[dataset]
    signal = cfg['signal']

    emb_p = embeddings_path(dataset, size, quality, algo)
    sig_p = signal_path(dataset, quality, signal, algo)

    if not emb_p.exists() or not sig_p.exists():
        return None

    suffix = '_geneformer' if algo == 'Geneformer' else ''
    signal_stem = f'Y_{signal}_{quality}{suffix}'
    out_path = ksg_output_path(dataset, size, quality, algo, signal_stem)

    if out_path.exists():
        return None  # already done

    try:
        X, Y = load_xy(dataset, emb_p, sig_p, quality)
        if X is None or Y is None:
            return {'dataset': dataset, 'algorithm': algo, 'size': size,
                    'quality': quality, '_error': 'load returned None'}
        n = X.shape[0]
        if n < KSG_K + 2:
            return {'dataset': dataset, 'algorithm': algo, 'size': size,
                    'quality': quality, '_error': f'too few samples ({n})'}

        pmi = ksg.mi(X, Y, k=KSG_K, base=2)
        mi_value = float(np.nanmean(pmi))
    except Exception as e:
        return {'dataset': dataset, 'algorithm': algo, 'size': size,
                'quality': quality, '_error': str(e)[:200]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f'{mi_value:.6f}')

    return {
        'dataset': dataset, 'algorithm': algo, 'size': size,
        'quality': quality, 'signal': signal, 'mi_value': mi_value,
    }


# ---------------------------------------------------------------------------
# Identify missing runs
# ---------------------------------------------------------------------------
def find_missing_jobs():
    all_jobs = []
    already_done = 0
    no_embeddings = 0

    for ds, cfg in EXPECTED.items():
        if ds not in ACTIVE_DATASETS:
            continue
        signal = cfg['signal']
        for algo, size, quality in product(ALGOS, cfg['sizes'], cfg['qualities']):
            emb_p = embeddings_path(ds, size, quality, algo)
            sig_p = signal_path(ds, quality, signal, algo)
            if not emb_p.exists() or not sig_p.exists():
                no_embeddings += 1
                continue

            suffix = '_geneformer' if algo == 'Geneformer' else ''
            signal_stem = f'Y_{signal}_{quality}{suffix}'
            if ksg_output_path(ds, size, quality, algo, signal_stem).exists():
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
        for ds, cfg in EXPECTED.items() if ds in ACTIVE_DATASETS
    )

    jobs, already_done, no_embeddings = find_missing_jobs()

    print(f'Total expected:    {total_expected}')
    print(f'Already done:      {already_done}')
    print(f'No embeddings/sig: {no_embeddings}')
    print(f'Missing (to run):  {len(jobs)}')
    print(f'Workers:           {N_WORKERS}')
    print(f'KSG params:        k={KSG_K}, base=2 (bits), no PCA, no sample cap')

    if not jobs:
        print('\nNothing to do — all KSG values are up to date.')
        sys.exit(0)

    results, errors = [], []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_ksg, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc='KSG'):
            r = future.result()
            if r is None:
                continue
            if '_error' in r:
                errors.append(r)
            else:
                results.append(r)

    df_results = pd.DataFrame(results)
    print(f'\nComputed {len(df_results)} KSG values, {len(errors)} errors')

    if errors:
        print('First errors:')
        for e in errors[:10]:
            print(f"  {e['dataset']}/{e['algorithm']}/{e['size']}/{e['quality']}: {e['_error']}")

    if len(df_results) > 0:
        pivot = df_results.groupby(['dataset', 'algorithm']).size().unstack(fill_value=0)
        print('\nNew KSG values per dataset x algorithm:')
        print(pivot)
