"""Compression variants, sparsity measurement, and saving/reloading inference artifacts."""
import copy
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.nn.quantized import dynamic as quantizeddynamic

from src import config
from src.model import MLP

# torch.ao.quantization prints deprecation warnings in recent PyTorch releases.
# The API still works; docs/decisions.md records why it is used.
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", message=".*quantized tensor creation functions.*")
warnings.filterwarnings("ignore", message=".*qnnpack incorrectly ignores reduce_range.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


def chooseengine():
    """Pick the best available INT8 CPU engine: x86/fbgemm on Intel or AMD, qnnpack on ARM."""
    supported = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No INT8 engine available: {supported}")


def variantnames():
    names = ["fp32"] + [f"pruned{round(level * 100)}" for level in config.prunelevels]
    return names + ["int8", f"pruned{round(config.combinedlevel * 100)}int8"]


def prunemodel(model, amount):
    """Global unstructured L1 pruning of every Linear weight on an independent copy.

    Biases are not pruned. prune.remove makes the zeros permanent, so the saved
    state dictionary has ordinary dense weight tensors that contain zeros.
    """
    pruned = copy.deepcopy(model)
    weights = [(layer, "weight") for layer in pruned.modules() if isinstance(layer, nn.Linear)]
    prune.global_unstructured(weights, pruning_method=prune.L1Unstructured, amount=amount)
    for layer, name in weights:
        prune.remove(layer, name)
    return pruned


def quantizemodel(model):
    """Dynamic INT8 quantization of Linear layers (INT8 weights, activations quantized at run time)."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu(), {nn.Linear},
                                                  dtype=torch.qint8)


def makevariants(baseline):
    """Every variant is derived from the same baseline, never from another variant."""
    variants = {"fp32": copy.deepcopy(baseline)}
    for level in config.prunelevels:
        variants[f"pruned{round(level * 100)}"] = prunemodel(baseline, level)
    variants["int8"] = quantizemodel(baseline)
    combined = prunemodel(baseline, config.combinedlevel)
    variants[f"pruned{round(config.combinedlevel * 100)}int8"] = quantizemodel(combined)
    return variants


def layerweights(model):
    """Return (layer name, float weight tensor) for float and dynamic-quantized Linear layers."""
    found = []
    for name, layer in model.named_modules():
        if isinstance(layer, quantizeddynamic.Linear):
            found.append((name, layer.weight().dequantize()))
        elif isinstance(layer, nn.Linear):
            found.append((name, layer.weight.detach()))
    return found


def sparsity(model):
    """Measured fraction of exactly-zero weights, overall and per layer."""
    rows = []
    for name, weight in layerweights(model):
        zeros = int((weight == 0).sum())
        deadunits = int((weight.abs().sum(dim=1) == 0).sum())
        rows.append({"layer": name, "weights": weight.numel(), "zeros": zeros,
                     "sparsity": zeros / weight.numel(), "deadunits": deadunits})
    total = sum(row["weights"] for row in rows)
    overall = sum(row["zeros"] for row in rows) / total
    return overall, rows


def saveartifact(model, path):
    """Save an inference-only state dictionary and return its size in bytes.

    Every artifact is called model.pt inside its own folder. torch.save writes the
    file name into the archive, so identical names keep sizes comparable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return os.path.getsize(path)


def loadartifact(name, path):
    """Rebuild the right architecture for a variant and load its saved weights."""
    model = MLP()
    if name.endswith("int8"):
        model = quantizemodel(model)
    model.load_state_dict(torch.load(path, weights_only=False))
    return model.eval()
