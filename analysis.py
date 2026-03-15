"""
Autoresearch Experiment Analysis.

Reads results.tsv and produces:
- Console summary of experiment outcomes
- progress.png chart of val_bpb over time
- Ranked list of top improvements

Usage: uv run analysis.py [--output-dir data/]
"""

import argparse
import glob
import os

import orjson
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path="results.tsv"):
    df = pd.read_csv(path, sep="\t")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    if "avg_tok_sec" in df.columns:
        df["avg_tok_sec"] = pd.to_numeric(df["avg_tok_sec"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()
    return df


def print_summary(df):
    print(f"Total experiments: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    counts = df["status"].value_counts()
    print("Experiment outcomes:")
    print(counts.to_string())

    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_decided = n_keep + n_discard
    if n_decided > 0:
        print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
    print()

    kept = df[df["status"] == "KEEP"].copy()
    has_tput = "avg_tok_sec" in df.columns and df["avg_tok_sec"].notna().any()
    print(f"KEPT experiments ({len(kept)} total):")
    for i, row in kept.iterrows():
        bpb = row["val_bpb"]
        desc = row["description"]
        tput_str = f"  tok/s={row['avg_tok_sec']:,.0f}" if has_tput and pd.notna(row.get("avg_tok_sec")) else ""
        print(f"  #{i:3d}  bpb={bpb:.6f}  mem={row['memory_gb']:.1f}GB{tput_str}  {desc}")
    print()


def print_stats(df):
    kept = df[df["status"] == "KEEP"].copy()
    baseline_bpb = df.iloc[0]["val_bpb"]
    best_bpb = kept["val_bpb"].min()
    best_row = kept.loc[kept["val_bpb"].idxmin()]

    print(f"Baseline val_bpb:  {baseline_bpb:.6f}")
    print(f"Best val_bpb:      {best_bpb:.6f}")
    print(f"Total improvement: {baseline_bpb - best_bpb:.6f} ({(baseline_bpb - best_bpb) / baseline_bpb * 100:.2f}%)")
    print(f"Best experiment:   {best_row['description']}")
    print()

    print("Cumulative effort per improvement:")
    kept_sorted = kept.reset_index()
    for _, row in kept_sorted.iterrows():
        desc = str(row["description"]).strip()
        print(f"  Experiment #{row['index']:3d}: bpb={row['val_bpb']:.6f}  {desc}")
    print()


def print_top_hits(df):
    kept = df[df["status"] == "KEEP"].copy()
    kept["prev_bpb"] = kept["val_bpb"].shift(1)
    kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]

    hits = kept.iloc[1:].copy()
    hits = hits.sort_values("delta", ascending=False)

    print(f"{'Rank':>4}  {'Delta':>8}  {'BPB':>10}  Description")
    print("-" * 80)
    for rank, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"{rank:4d}  {row['delta']:+.6f}  {row['val_bpb']:.6f}  {row['description']}")

    print(f"\n{'':>4}  {hits['delta'].sum():+.6f}  {'':>10}  TOTAL improvement over baseline")
    print()


def plot_progress(df, output_path="progress.png"):
    has_tput = "avg_tok_sec" in df.columns and df["avg_tok_sec"].notna().any()
    nrows = 2 if has_tput else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 8 * nrows),
                             sharex=True, squeeze=False)
    ax = axes[0, 0]

    valid = df[df["status"] != "CRASH"].copy()
    valid = valid.reset_index(drop=True)

    baseline_bpb = valid.loc[0, "val_bpb"]

    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

    disc = below[below["status"] == "DISCARD"]
    ax.scatter(disc.index, disc["val_bpb"],
               c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    kept_v = below[below["status"] == "KEEP"]
    ax.scatter(kept_v.index, kept_v["val_bpb"],
               c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)

    kept_mask = valid["status"] == "KEEP"
    kept_idx = valid.index[kept_mask]
    kept_bpb = valid.loc[kept_mask, "val_bpb"]
    running_min = kept_bpb.cummin()
    ax.step(kept_idx, running_min, where="post", color="#27ae60",
            linewidth=2, alpha=0.7, zorder=3, label="Running best")

    for idx, bpb in zip(kept_idx, kept_bpb):
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."
        ax.annotate(desc, (idx, bpb),
                    textcoords="offset points",
                    xytext=(6, 6), fontsize=8.0,
                    color="#1a7a3a", alpha=0.9,
                    rotation=30, ha="left", va="bottom")

    best = kept_bpb.min()
    n_total = len(df)
    n_kept = len(df[df["status"] == "KEEP"])
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    margin = (baseline_bpb - best) * 0.15
    ax.set_ylim(best - margin, baseline_bpb + margin)

    # Throughput subplot
    if has_tput:
        ax2 = axes[1, 0]
        tput_valid = valid[valid["avg_tok_sec"].notna()]
        disc_t = tput_valid[tput_valid["status"] == "DISCARD"]
        ax2.scatter(disc_t.index, disc_t["avg_tok_sec"],
                    c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")
        kept_t = tput_valid[tput_valid["status"] == "KEEP"]
        ax2.scatter(kept_t.index, kept_t["avg_tok_sec"],
                    c="#3498db", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)
        if len(kept_t) > 0:
            ax2.step(kept_t.index, kept_t["avg_tok_sec"], where="post",
                     color="#2980b9", linewidth=2, alpha=0.7, zorder=3, label="Kept trend")
        ax2.set_ylabel("Avg tok/sec (higher is better)", fontsize=12)
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Experiment #", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved progress chart to {output_path}")


def load_runs(data_dir="data"):
    """Read all data/run_*.json files, return list of dicts sorted by timestamp."""
    pattern = os.path.join(data_dir, "run_*.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            runs.append(orjson.loads(f.read()))
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs


def print_run_details(runs):
    """Print detailed cross-run comparison from JSON files."""
    if not runs:
        print("No run_*.json files found in data/.")
        return

    header = f"{'Timestamp':>19s}  {'Dataset':>10s}  {'Depth':>5s}  {'val_bpb':>8s}  {'tok/sec':>8s}  {'Steps':>5s}  {'TrainMB':>7s}  {'EvalSec':>7s}  {'Compiled':>8s}  {'Batch':>5s}"
    print(header)
    print("-" * len(header))

    for run in runs:
        ts = run.get("timestamp", "?")
        dataset = run.get("data", {}).get("dataset", "?")
        depth = run.get("model", {}).get("depth", "?")
        val_bpb = run.get("result", {}).get("val_bpb", 0)
        t = run.get("training", {})
        tok_sec = t.get("avg_tok_sec", 0)
        steps = t.get("total_steps", 0)
        train_mb = t.get("training_peak_mb", 0)
        eval_sec = t.get("eval_seconds", "")
        compiled = "yes" if t.get("compiled") else "no"
        batch = t.get("batch_size", "?")

        eval_str = f"{eval_sec:.1f}" if isinstance(eval_sec, (int, float)) else "-"
        print(f"{ts:>19s}  {str(dataset):>10s}  {str(depth):>5s}  {val_bpb:8.6f}  {tok_sec:>8,}  {steps:5d}  {train_mb:7.0f}  {eval_str:>7s}  {compiled:>8s}  {str(batch):>5s}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze autoresearch experiment results")
    parser.add_argument("--results", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--output-dir", default="data", help="Directory for output files")
    args = parser.parse_args()

    # Detailed view from run_*.json files
    runs = load_runs(args.output_dir)
    if runs:
        print("=== Detailed Run History (from run_*.json) ===\n")
        print_run_details(runs)

    # Summary from results.tsv
    if os.path.exists(args.results):
        df = load_results(args.results)

        print("=== Experiment Summary (from results.tsv) ===\n")
        print_summary(df)
        print_stats(df)
        print_top_hits(df)

        os.makedirs(args.output_dir, exist_ok=True)
        plot_progress(df, output_path=os.path.join(args.output_dir, "progress.png"))
    else:
        print(f"No {args.results} found -- skipping TSV summary.")
