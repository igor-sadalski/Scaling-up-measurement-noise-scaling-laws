"""Compute MI on 20 evenly-spaced Geneformer checkpoints per (size, quality) for shendure.

Each (size, quality) pair is handled sequentially inside one worker to avoid races on
the shared embeddings.csv (written by EmbExtractor into results/Geneformer/) and the
shared test-signal CSV (written by Geneformer.embed). Pairs are dispatched across
workers in a ProcessPoolExecutor; each worker randomly picks a GPU at startup.

Outputs land at the per-checkpoint path:
  data/shendure/<size>/<quality>/results/Geneformer/<checkpoint-NNNN>/MI/42/
  Y_author_day_<quality>_geneformer/lmi_mutual_information.txt
"""

import os
import sys
import json
import random
import threading
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


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


def _setup_main_logging() -> None:
    """Only the main process should tee stdout/stderr into the shared log.
    Spawn workers re-import this module and would otherwise truncate the log."""
    log_f = open(LOG_PATH, "w")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"Logging to {LOG_PATH}")


DATA_ROOT = Path("/home/igor/noise_scaling/data/shendure")
DATASET_NAME = "shendure"
SIGNAL_COLUMNS = ["author_day"]
SEED = 42
N_CHECKPOINTS = 20
MAX_WORKERS = 32
N_GPUS = 8
INFERENCE_BATCH_SIZE = 50
MI_MAX_EPOCHS = 300
WORKER_LOG_DIR = SCRIPT_PATH.parent / f"{SCRIPT_PATH.stem}_worker_logs"


def list_sizes(root: Path) -> list[str]:
    return sorted(
        (p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=int,
    )


def list_qualities(size_dir: Path) -> list[str]:
    out = []
    for p in size_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            float(p.name)
            out.append(p.name)
        except ValueError:
            continue
    return sorted(out, key=float)


def list_checkpoints(results_geneformer_dir: Path) -> list[str]:
    ckpts = []
    for p in results_geneformer_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
            except ValueError:
                continue
            if (p / "config.json").exists() and any(p.glob("model.safetensors*")):
                ckpts.append((step, p.name))
    ckpts.sort(key=lambda t: t[0])
    return [name for _, name in ckpts]


def pick_evenly_spaced(names: list[str], k: int) -> list[str]:
    if not names:
        return []
    if len(names) <= k:
        return names
    idx = np.unique(np.linspace(0, len(names) - 1, k).round().astype(int))
    return [names[i] for i in idx]


def mi_txt_path(base_dir: Path, ckpt_name: str, quality: str, seed: int) -> Path:
    signal_stem = f"Y_{SIGNAL_COLUMNS[0]}_{quality}_geneformer"
    return (
        base_dir
        / "results"
        / "Geneformer"
        / ckpt_name
        / "MI"
        / str(seed)
        / signal_stem
        / "lmi_mutual_information.txt"
    )


def already_done(base_dir: Path, ckpt_name: str, quality: str) -> bool:
    p = mi_txt_path(base_dir, ckpt_name, quality, SEED)
    if not p.exists():
        return False
    try:
        float(p.read_text().strip())
        return True
    except Exception:
        return False


def build_pair_tasks() -> list[dict]:
    tasks = []
    for size in list_sizes(DATA_ROOT):
        size_dir = DATA_ROOT / size
        for quality in list_qualities(size_dir):
            base_dir = size_dir / quality
            results_gf = base_dir / "results" / "Geneformer"
            if not results_gf.is_dir():
                continue
            test_tok = DATA_ROOT / "test" / quality / "preprocessed" / "tokenized.dataset"
            token_dict = DATA_ROOT / "utils" / "token_dict.pkl"
            if not test_tok.exists() or not token_dict.exists():
                continue
            ckpts = list_checkpoints(results_gf)
            if not ckpts:
                continue
            picked = pick_evenly_spaced(ckpts, N_CHECKPOINTS)
            pending = [c for c in picked if not already_done(base_dir, c, quality)]
            tasks.append(
                {
                    "size": size,
                    "quality": quality,
                    "checkpoints_picked": picked,
                    "checkpoints_pending": pending,
                }
            )
    return tasks


def process_size_quality(
    size: str, quality: str, ckpt_names: list[str], gpu_id: int, progress_queue
) -> dict:
    """Runs inside a fresh spawn subprocess — set CUDA_VISIBLE_DEVICES before torch imports."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")

    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    worker_log = WORKER_LOG_DIR / f"{size}_{quality}.log"
    wf = open(worker_log, "w", buffering=1)
    sys.stdout = wf
    sys.stderr = wf
    print(f"[worker] size={size} quality={quality} gpu={gpu_id} ckpts={len(ckpt_names)}")

    from scaling_laws.algo.geneformer import Geneformer  # noqa: E402

    base_dir = DATA_ROOT / size / quality
    results = []
    for ckpt in ckpt_names:
        if already_done(base_dir, ckpt, quality):
            p = mi_txt_path(base_dir, ckpt, quality, SEED)
            try:
                mi_val = float(p.read_text().strip())
            except Exception:
                mi_val = None
            results.append({"checkpoint": ckpt, "mi": mi_val, "status": "skipped", "error": None})
            print(f"  [skip] {ckpt} already has MI file")
            progress_queue.put(
                {"size": size, "quality": quality, "ckpt": ckpt, "status": "skipped", "mi": mi_val}
            )
            continue
        try:
            print(f"  [run ] {ckpt}")
            gf = Geneformer(
                base_dir=str(base_dir),
                signal_columns=SIGNAL_COLUMNS,
                device=gpu_id,
                dataset_name=DATASET_NAME,
                model_name=ckpt,
                seed=SEED,
            )
            gf.embed(inference_batch_size=INFERENCE_BATCH_SIZE)
            mi_map = gf.mutual_information(max_epochs=MI_MAX_EPOCHS)
            p = mi_txt_path(base_dir, ckpt, quality, SEED)
            mi_val = float(p.read_text().strip()) if p.exists() else None
            results.append(
                {
                    "checkpoint": ckpt,
                    "mi": mi_val,
                    "status": "ok",
                    "error": None,
                    "raw_mi_map": {k: float(v) for k, v in mi_map.items()},
                }
            )
            print(f"  [done] {ckpt} mi={mi_val}")
            progress_queue.put(
                {"size": size, "quality": quality, "ckpt": ckpt, "status": "ok", "mi": mi_val}
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [ERR ] {ckpt}: {e}\n{tb}")
            results.append(
                {"checkpoint": ckpt, "mi": None, "status": "error", "error": str(e)}
            )
            progress_queue.put(
                {"size": size, "quality": quality, "ckpt": ckpt, "status": "error", "mi": None}
            )

    return {"size": size, "quality": quality, "gpu_id": gpu_id, "results": results}


def main():
    _setup_main_logging()
    print(f"Scanning {DATA_ROOT} for (size, quality) pairs...")
    tasks = build_pair_tasks()
    total_pairs = len(tasks)
    total_ckpts = sum(len(t["checkpoints_picked"]) for t in tasks)
    total_pending = sum(len(t["checkpoints_pending"]) for t in tasks)
    print(
        f"Found {total_pairs} (size, quality) pairs; "
        f"{total_ckpts} checkpoint jobs picked; {total_pending} pending (not already done)."
    )

    work = [t for t in tasks if t["checkpoints_pending"]]
    if not work:
        print("Nothing to do — all picked checkpoints already have MI results.")
        return

    random.seed(0)
    random.shuffle(work)
    total_jobs = sum(len(t["checkpoints_pending"]) for t in work)
    print(
        f"Dispatching {len(work)} (size, quality) pair(s) "
        f"= {total_jobs} checkpoint MI jobs across {MAX_WORKERS} workers on {N_GPUS} GPUs."
    )

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    progress_queue = manager.Queue()

    counters = {"ok": 0, "skipped": 0, "error": 0}
    pbar = tqdm(total=total_jobs, desc="checkpoint MI jobs", unit="ckpt", smoothing=0.1)
    stop_sentinel = "__STOP__"

    def _consume_progress():
        while True:
            msg = progress_queue.get()
            if msg == stop_sentinel:
                return
            status = msg.get("status", "ok")
            counters[status] = counters.get(status, 0) + 1
            mi_str = f"{msg['mi']:.3f}" if isinstance(msg.get("mi"), float) else "NA"
            pbar.set_postfix(
                ok=counters["ok"],
                skip=counters["skipped"],
                err=counters["error"],
                last=f"{msg['size']}/{msg['quality']}/{msg['ckpt']}={mi_str}",
                refresh=False,
            )
            pbar.update(1)

    consumer = threading.Thread(target=_consume_progress, daemon=True)
    consumer.start()

    all_results = []
    summary_rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futures = {}
        for t in work:
            gpu_id = random.randint(0, N_GPUS - 1)
            fut = ex.submit(
                process_size_quality,
                t["size"],
                t["quality"],
                t["checkpoints_pending"],
                gpu_id,
                progress_queue,
            )
            futures[fut] = (t["size"], t["quality"], gpu_id)

        pair_bar = tqdm(total=len(futures), desc="(size, quality) pairs", position=1, leave=True)
        for fut in as_completed(futures):
            size, quality, gpu_id = futures[fut]
            try:
                res = fut.result()
                all_results.append(res)
                n_ok = sum(1 for r in res["results"] if r["status"] == "ok")
                n_err = sum(1 for r in res["results"] if r["status"] == "error")
                n_skip = sum(1 for r in res["results"] if r["status"] == "skipped")
                tqdm.write(
                    f"[main] size={size} quality={quality} gpu={gpu_id} "
                    f"ok={n_ok} skipped={n_skip} err={n_err}"
                )
                for r in res["results"]:
                    summary_rows.append(
                        {
                            "size": size,
                            "quality": quality,
                            "checkpoint": r["checkpoint"],
                            "mi": r["mi"],
                            "status": r["status"],
                            "error": r["error"],
                        }
                    )
            except Exception as e:
                tb = traceback.format_exc()
                tqdm.write(f"[main] FATAL size={size} quality={quality} gpu={gpu_id}: {e}\n{tb}")
                summary_rows.append(
                    {
                        "size": size,
                        "quality": quality,
                        "checkpoint": None,
                        "mi": None,
                        "status": "worker_fatal",
                        "error": str(e),
                    }
                )
            pair_bar.update(1)
        pair_bar.close()

    progress_queue.put(stop_sentinel)
    consumer.join(timeout=10)
    pbar.close()

    summary_csv = SCRIPT_PATH.with_suffix(".csv")
    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    summary_json = SCRIPT_PATH.with_suffix(".json")
    summary_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote full JSON to {summary_json}")


if __name__ == "__main__":
    main()
