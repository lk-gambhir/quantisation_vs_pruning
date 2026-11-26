"""CPU latency benchmark. Only the forward pass is timed."""
import time

import numpy as np
import torch

from src import config


def measure(name, seed, model, images):
    """Time one model on fixed real test images for each configured batch size.

    Returns one row per timed iteration so p50/p95/IQR can be recomputed later.
    """
    torch.set_num_threads(config.benchthreads)
    model = model.cpu().eval()
    rows = []
    with torch.inference_mode():
        for batchsize in config.benchbatches:
            batch = images[:batchsize].contiguous()
            for _ in range(config.warmup):
                model(batch)
            for iteration in range(1, config.repeats + 1):
                start = time.perf_counter()
                model(batch)
                elapsed = (time.perf_counter() - start) * 1000
                rows.append({"seed": seed, "variant": name, "batchsize": batchsize,
                             "iteration": iteration, "latencyms": elapsed})
    return rows


def summarize(times):
    """p50, p95 and interquartile range (all in milliseconds) for one list of timings."""
    q1, p50, q3, p95 = np.percentile(times, [25, 50, 75, 95])
    return {"p50ms": p50, "p95ms": p95, "iqrms": q3 - q1}
