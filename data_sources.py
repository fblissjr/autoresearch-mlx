"""
Dataset registry and configuration for autoresearch.

Supports multiple datasets with per-dataset cache directories, tokenizer
settings, and download strategies. Keeps prepare.py aligned with upstream
by isolating all dataset-switching logic here.

Usage:
    from data_sources import configure_dataset
    ds = configure_dataset("tinystories")  # updates prepare.py globals, returns config
"""

import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests

import prepare

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "climbmix": {
        "hf_repo": "karpathy/climbmix-400b-shuffle",
        "max_shard": 6542,
        "val_shard": 6542,
        "vocab_size": 8192,
        "max_seq_len": 2048,
        "eval_tokens": 40 * 524288,
        "type": "pre_sharded",
    },
    "tinystories": {
        "hf_repo": "karpathy/tinystories-gpt4-clean",
        "vocab_size": 2048,
        "max_seq_len": 512,
        "eval_tokens": 3 * 524288,
        "type": "single_dataset",
        "num_local_shards": 21,  # 20 train + 1 val
    },
}

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

def configure_dataset(name="climbmix"):
    """
    Configure prepare.py module globals for the selected dataset.
    Returns the dataset config dict for callers that need values directly.
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    ds = DATASETS[name]

    # Per-dataset cache directories (climbmix keeps legacy flat path)
    if name == "climbmix":
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", name)

    prepare.CACHE_DIR = cache_dir
    prepare.DATA_DIR = os.path.join(cache_dir, "data")
    prepare.TOKENIZER_DIR = os.path.join(cache_dir, "tokenizer")

    prepare.VOCAB_SIZE = ds["vocab_size"]
    prepare.MAX_SEQ_LEN = ds["max_seq_len"]
    prepare.EVAL_TOKENS = ds["eval_tokens"]

    if ds["type"] == "pre_sharded":
        prepare.MAX_SHARD = ds["max_shard"]
        prepare.VAL_SHARD = ds["val_shard"]
        prepare.VAL_FILENAME = f"shard_{ds['val_shard']:05d}.parquet"
        prepare.BASE_URL = f"https://huggingface.co/datasets/{ds['hf_repo']}/resolve/main"
    else:
        num_shards = ds.get("num_local_shards", 21)
        prepare.MAX_SHARD = num_shards - 1
        prepare.VAL_SHARD = num_shards - 1
        prepare.VAL_FILENAME = f"shard_{num_shards - 1:05d}.parquet"
        prepare.BASE_URL = None

    return ds


# ---------------------------------------------------------------------------
# Download helpers for non-pre-sharded datasets
# ---------------------------------------------------------------------------

def _fetch_parquet_urls(hf_repo):
    """Fetch parquet file URLs from the HF dataset API."""
    api_url = f"https://huggingface.co/api/datasets/{hf_repo}/parquet"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    info = response.json()

    urls = []
    # Handle nested split format: {"default": {"train": [...]}} or {"train": [...]}
    def _extract_urls(obj):
        if isinstance(obj, str) and obj.startswith("http"):
            urls.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract_urls(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_urls(item)

    _extract_urls(info)
    if not urls:
        raise RuntimeError(f"No parquet files found for {hf_repo}")
    return urls


def download_and_shard_dataset(hf_repo, data_dir, num_shards=21):
    """Download a HF dataset and shard it locally into parquet files."""
    os.makedirs(data_dir, exist_ok=True)

    # Check if already sharded
    existing = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("shard_") and f.endswith(".parquet")
    )
    if len(existing) >= num_shards:
        print(f"Data: all {len(existing)} shards already exist at {data_dir}")
        return

    print(f"Data: fetching parquet URLs for {hf_repo}...")
    parquet_urls = _fetch_parquet_urls(hf_repo)
    print(f"Data: found {len(parquet_urls)} parquet file(s)")

    # Download to temp directory
    temp_dir = os.path.join(data_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    downloaded = []
    for i, url in enumerate(parquet_urls):
        temp_path = os.path.join(temp_dir, f"part_{i:03d}.parquet")
        if not os.path.exists(temp_path):
            print(f"  Downloading part {i + 1}/{len(parquet_urls)}...")
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = requests.get(url, stream=True, timeout=60)
                    resp.raise_for_status()
                    with open(temp_path + ".tmp", "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    os.rename(temp_path + ".tmp", temp_path)
                    break
                except (requests.RequestException, IOError) as e:
                    print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
                    for path in [temp_path + ".tmp", temp_path]:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass
                    if attempt < max_attempts:
                        time.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(f"Failed to download {url}") from e
        downloaded.append(temp_path)

    # Read all texts
    print("  Reading downloaded data...")
    all_texts = []
    for path in downloaded:
        table = pq.read_table(path, columns=["text"])
        all_texts.extend(table.column("text").to_pylist())

    print(f"  Total documents: {len(all_texts):,}")

    # Deterministic shuffle and split into shards
    import random
    rng = random.Random(42)
    rng.shuffle(all_texts)

    docs_per_shard = len(all_texts) // num_shards
    for shard_idx in range(num_shards):
        start = shard_idx * docs_per_shard
        end = start + docs_per_shard if shard_idx < num_shards - 1 else len(all_texts)
        shard_texts = all_texts[start:end]

        table = pa.table({"text": shard_texts})
        shard_path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path)
        print(f"  Wrote shard_{shard_idx:05d}.parquet ({len(shard_texts):,} docs)")

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Data: {num_shards} shards ready at {data_dir}")
