# Project overview

## Title

**How Much Can We Compress an Image Classifier? A Controlled Comparison of Magnitude Pruning and INT8 Quantization**

## Status

The core study is complete for three paired seeds (42, 123, 2026), which is the protocol minimum. `notebooks/start.ipynb` runs the whole pipeline from a fresh environment, and results are in `results.md`. Two earlier single-seed pilots (a remote GPU server and a laptop notebook) are kept for the record, but only the three-seed run is used for conclusions. Open items are listed in `tasks.md`: five seeds, teacher requirements check, report and presentation.

## One-sentence summary

Train one small classifier per seed, derive every compressed variant from that same trained model, and measure accuracy, saved file size and CPU latency separately, so that "has zeros", "is smaller" and "is faster" are not mixed up.

## Research question

For a fixed Fashion-MNIST classifier, which of unstructured magnitude pruning and dynamic INT8 quantization gives the best measured storage and CPU-latency trade-off at a predeclared accuracy limit? Do the zeros in a dense pruned model turn into actual storage or inference gains?

## Why this is a useful study

Compression claims often mix up three different outcomes: zero weights, smaller files and faster execution. This project measures each one separately on the same model and machine. A result where pruning creates many zeros but does not reduce file size or latency is a valid finding, not a failed experiment.

## Scope

### In scope (done)

- Fashion-MNIST with the official training and test partitions.
- One MLP (`784 → 256 → 128 → 10`) as the only architecture.
- An FP32 baseline per seed, and compressed variants derived from it.
- Global unstructured L1 magnitude pruning at 30%, 50% and 70% (core) and 90% (added to locate where accuracy collapses; see `decisions.md`).
- Dynamic INT8 quantization of the `Linear` layers.
- A combined variant: 50% pruning, then dynamic INT8.
- Accuracy, macro F1, per-class precision/recall/F1, confusion matrices, measured sparsity (overall and per layer), saved artifact bytes and bits per parameter, and CPU latency (p50, p95, IQR, with every raw timing saved) at batch sizes 1 and 64.
- Three paired seeds, with mean ± SD and paired deltas.
- Diagnostics: per-layer weight distributions with pruning cut-offs, per-layer sparsity and dead neurons, per-class F1 changes, example-level failure IDs, and the prediction collapse at 90%.

### Possible extensions (not done; discuss with the teacher first)

- Seeds 2027 and 2028 to reach the five-seed target (a one-line change in `src/config.py`).
- Fine-tuning after pruning, reported as a separate, clearly labelled experiment.
- Structured (neuron) pruning, which physically shrinks the matrices.
- A sparse storage format (for example CSR), with its index overhead included in the size.
- Latency on an x86 CPU with the fbgemm/x86 INT8 engine.

### Out of scope

- Deployment claims for phones, wearables or microcontrollers that were not tested.
- Other architectures, datasets or compression libraries.
- Static post-training quantization, quantization-aware training, INT4, ONNX/FastAPI serving, and CIFAR-10/ResNet.
- Claiming pruning is faster just because weights are zero.
- Designing a new compression algorithm.

## Model and data

- Dataset: Fashion-MNIST, 28×28 grayscale, 10 balanced classes (6,000 train and 1,000 test images per class).
- Input: `ToTensor` scaling to [0, 1], then flattened to 784 values. No other normalisation.
- MLP: `784 → 256 → 128 → 10` with ReLU between hidden layers. It outputs logits and has 235,146 parameters.
- Training: cross-entropy, Adam with learning rate `1e-3`, batch size 128, 10 epochs. The checkpoint kept is the epoch with the lowest validation loss.
- Split: 54,000 train and 6,000 validation, taken from the official training partition with a seeded generator for each seed. The official test set is used only for final evaluation.

## Comparison groups

1. FP32 baseline.
2. Dynamic INT8.
3. Magnitude pruning at 30%, 50%, 70% and 90%.
4. 50% pruning followed by dynamic INT8.

The comparison is paired: within a seed, every variant starts from a copy of that seed's baseline.

## Decision criterion (pre-registered)

Choose the smallest saved artifact whose mean accuracy is within **1 percentage point** of the paired FP32 baseline, then compare latency. A nominal sparsity level is not treated as a size budget. Result: dynamic INT8 (see `results.md`).

## Contribution

This is a reproducible, controlled case study of whether pruning-versus-quantization trade-offs reported in prior work hold for a small MLP when storage and CPU latency are actually measured. It separates dense zero sparsity from file compression, repeats every comparison across seeds, and analyses which classes and layers are affected. It is not a new algorithm or a universal result.

## Deliverables

| Deliverable | Where |
|---|---|
| Reproducible code and environment record | `src/`, `notebooks/start.ipynb`, `requirements.txt`, `outputs/setup.json` |
| Dataset and split documentation | `protocol.md`, `outputs/dataset-summary.json`, `outputs/runs/seed*/split.json` |
| Saved inference artifacts (recreated by running the notebook) | `outputs/checkpoints/seed*/<variant>/model.pt` |
| Machine-readable metrics and run configuration | `outputs/metrics/`, `outputs/runs/config.json` |
| Required figures (quality vs size, quality vs latency, quality and sparsity vs pruning level) | `outputs/figures/` |
| Diagnostics (weight distributions, per-class and example failures) | `outputs/figures/`, `outputs/metrics/failures-seed42.csv` |
| Results, analysis and limitations | `results.md` |
| Final report and presentation in the teacher's format | To do (`tasks.md` T52–T53) |
