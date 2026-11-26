"""Shared settings for every experiment step.

Change values here, not inside the notebook, so that every seed and every
variant uses exactly the same procedure. The values below are the frozen
protocol described in docs/protocol.md.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
datadir = root / "data"
outdir = root / "outputs"

# Repeated runs: three paired seeds (protocol minimum). Add 2027 and 2028 for the five-seed target.
seeds = [42, 123, 2026]

# Data and baseline training
valfraction = 0.1
batchsize = 128
learningrate = 1e-3
epochs = 10
hidden = [256, 128]

# Compression variants
prunelevels = [0.3, 0.5, 0.7, 0.9]
combinedlevel = 0.5

# Decision rule: largest allowed mean accuracy drop, in percentage points
tolerance = 1.0

# CPU latency benchmark
benchthreads = 1
benchbatches = [1, 64]
warmup = 30
repeats = 200


def settings():
    """Return the settings as a plain dictionary so they can be saved with each run."""
    return {
        "seeds": seeds,
        "valfraction": valfraction,
        "batchsize": batchsize,
        "learningrate": learningrate,
        "epochs": epochs,
        "hidden": hidden,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "checkpointrule": "lowest validation loss",
        "prunelevels": prunelevels,
        "combinedlevel": combinedlevel,
        "tolerance": tolerance,
        "benchthreads": benchthreads,
        "benchbatches": benchbatches,
        "warmup": warmup,
        "repeats": repeats,
    }
