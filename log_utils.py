"""
Project-wide logging, diagnostics, and structured output for autoresearch-mlx.

Usage:
    from log_utils import logger, is_debug
    from log_utils import sample_memory, format_step_timings
    from log_utils import hardware_info, save_json, build_bench_data, FORMAT_VERSION

Enable debug mode by passing --debug flag to any script, or setting
the AUTORESEARCH_DEBUG=1 environment variable.
"""

import logging
import os
import platform
import sys
import time

import mlx.core as mx
import orjson

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _check_debug():
    """Check if debug mode is enabled via --debug flag or env var."""
    if os.environ.get("AUTORESEARCH_DEBUG", "0") == "1":
        return True
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        return True
    return False


is_debug = _check_debug()

logger = logging.getLogger("autoresearch")
logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if is_debug else logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Memory diagnostics
# ---------------------------------------------------------------------------

def sample_memory(step, interval=10):
    """Sample active and peak memory every `interval` steps.

    Returns (active_mb, peak_mb) or (None, None) if not a sampling step.
    """
    if step % interval != 0:
        return None, None
    active_mb = round(mx.get_active_memory() / 1024 / 1024, 1)
    peak_mb = round(mx.get_peak_memory() / 1024 / 1024, 1)
    return active_mb, peak_mb


def format_step_timings(step_timings):
    """Convert step_timings tuples to JSON-serializable dicts.

    Each tuple: (step, dt, tok_sec, loss, active_mb, peak_mb).
    Omits memory fields when None (non-sampling steps).
    """
    result = []
    for s, dt, ts, l, am, pm in step_timings:
        entry = {"step": s, "dt": dt, "tok_sec": ts, "loss": l}
        if am is not None:
            entry["active_mb"] = am
            entry["peak_mb"] = pm
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Structured JSON output (format_version 0.1)
# ---------------------------------------------------------------------------

FORMAT_VERSION = "0.1"


def hardware_info():
    """Return hardware metadata dict."""
    return {
        "chip": platform.processor() or "Apple Silicon",
        "memory_gb": None,
        "os": platform.system(),
    }


def save_json(prefix, data):
    """Write data to data/<prefix>_<timestamp>.json and print the path.

    Returns the output path.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("data", f"{prefix}_{timestamp}.json")
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"Results saved to {out_path}")
    return out_path


def build_bench_data(configs):
    """Build the structured bench JSON dict."""
    return {
        "format_version": FORMAT_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": hardware_info(),
        "configs": configs,
    }
