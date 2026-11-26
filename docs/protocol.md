# Experiment protocol

This document defines the rules for a fair comparison. It was frozen before the final three-seed run; the implemented values are in `src/config.py`. Any change must be logged in `decisions.md`, with its date and reason, before its results are treated as final.

## 1. Dataset and split

- Fashion-MNIST from TorchVision, using the official partitions (60,000 train, 10,000 test).
- Preprocessing: `ToTensor` only, which gives float pixels in [0, 1]. It is identical for every model.
- The official test set is not used for training, checkpoint selection, pruning choices or quantization. The seed-42 test set was seen in the pilots (`progress.md`), and this is disclosed in `results.md`. No setting was changed afterwards because of test results.
- For each seed, the official training partition is split 90% train (54,000) and 10% validation (6,000). A `torch.Generator` seeded with that seed's value permutes the indices, giving the same split as `random_split`. The exact indices are saved to `outputs/runs/seed<N>/split.json`.
- Class counts, image shape and pixel range are recorded in `outputs/dataset-summary.json`.

## 2. Seeds and pairing

- Completed seeds: `42`, `123` and `2026`, the minimum of three. The target of five adds `2027` and `2028`. Results state the actual count.
- Each seed has its own split, shuffling order, initial weights and trained baseline.
- Within a seed, every compressed variant is derived independently from that seed's baseline and evaluated on the same test examples. Variants are never derived from each other, except that the combined variant is pruned first and then quantized.

## 3. Baseline training

- Architecture: Flatten → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 10), with logits as output.
- Loss: cross-entropy. Optimizer: Adam, learning rate 0.001. Batch size 128. 10 epochs.
- Seeds: Python, NumPy and PyTorch are seeded before the model is created; the training loader uses its own seeded generator.
- Checkpoint: the epoch with the lowest validation loss. Validation accuracy is also reported.
- Training and validation loss and accuracy are saved for every epoch.

## 4. Compression methods

### FP32 control

The selected baseline, uncompressed. It is the reference for all deltas.

### Magnitude pruning

- Global unstructured L1 pruning (`torch.nn.utils.prune.global_unstructured` with `L1Unstructured`) over the weights of all three `Linear` layers. Biases are not pruned.
- Levels: 30%, 50%, 70% (core) and 90% (extension; see `decisions.md`). Each level is applied to a fresh copy of the baseline.
- No fine-tuning after pruning, so the effect of pruning alone can be measured.
- `prune.remove` makes the zeros permanent before saving.
- The implementation was checked by hand at 90%: the 211,277 weights with the smallest |w| exactly match the weights that PyTorch sets to zero.

### Dynamic INT8 quantization

- `torch.ao.quantization.quantize_dynamic` with `{nn.Linear}` and `dtype=torch.qint8`, applied to a CPU copy of the baseline. Weights are stored as INT8 with a per-tensor affine scale; activations are quantized at run time.
- Engine: the first available of `x86`, `fbgemm` and `qnnpack`. The engine used is recorded in `outputs/setup.json` (`qnnpack` on Apple M2).

### Combined variant

- 50% pruning, then dynamic INT8, on an independent copy. It is included because saving, reloading, prediction and timing all work in the same runtime.

## 5. Metrics

### Predictive quality

- Accuracy and macro F1 on all 10,000 test images.
- Per-class precision, recall and F1, and the confusion matrix.
- Paired deltas from the same seed's FP32 baseline, in percentage points.
- Example-level changes against FP32: new errors, fixed errors, both wrong, and any disagreement.

### Sparsity and storage

- Measured sparsity: the fraction of exactly-zero weights over all `Linear` weights, overall and per layer. For INT8 models the weights are dequantized before counting. Dead neurons (rows with all weights zero) are counted.
- Artifact bytes come from the saved file (`os.path.getsize`). Also reported: the compression ratio (FP32 bytes ÷ variant bytes, same seed) and bits per parameter (bytes × 8 ÷ 235,146 total parameters, including format overhead).
- Every artifact uses the same file name (`model.pt`) so that archive-name overhead is identical.
- Only dense storage is measured. A pruning target is not a storage budget, and zeros are not called "file compression" unless the file actually shrank.

### CPU latency

- CPU only, in `eval()` mode under `torch.inference_mode()`, with 1 thread (`torch.set_num_threads(1)`).
- Input: the first 64 test images (real data), sliced to batch sizes 1 and 64.
- For each seed, variant and batch size: 30 warm-up passes, then 200 timed passes. Only the forward pass is inside the `time.perf_counter()` timer.
- Every timing is saved. Each run is summarised as p50, p95 and IQR, and the cross-seed summary is the mean of the per-run values.
- Same machine, same code and same input for every variant. The machine is recorded in `outputs/setup.json`.

## 6. Statistical summary

Mean and sample SD across seeds for accuracy, F1 and paired deltas. With three seeds, no significance tests are claimed. Individual test images are not treated as independent runs.

## 7. Decision rule (pre-registered)

Among variants whose mean accuracy is within 1 pp of FP32, pick the one with the smallest saved artifact, then report its latency. This rule was not changed after seeing results.

## 8. Figures

Required: quality vs saved size (seed dots plus mean ± SD, with the 1 pp band marked), quality vs latency for both batch sizes, and quality and measured sparsity vs pruning target. Additional: learning curves, FP32 and 90% confusion matrices, per-class F1, the weight distribution with pruning cut-offs, per-layer sparsity, predicted-class counts, and a failure-example grid. Every figure is drawn from saved tables.

## 9. Diagnostics

- Per-layer |weight| histograms for one baseline (seed 42), with each global pruning cut-off marked.
- Per-class F1 deltas averaged across seeds.
- Every test image that FP32 classifies correctly but a variant gets wrong is saved by ID. The image grid shows the first 12 in test-index order, so the examples are not hand-picked.
- These diagnostics explain results; they do not replace the cross-seed statistics.

## 10. Interpretation rules

- Report observed values. Do not assume INT8 is four times smaller or faster.
- Do not call zero weights "compression" unless the file shrank, or "speed-up" unless the benchmark shows one.
- Do not claim one method is generally better based on one dataset, model and machine.
- Report negative or neutral outcomes, such as pruning not shrinking files or INT8 being slower, as findings.
