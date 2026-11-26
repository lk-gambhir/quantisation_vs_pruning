# Results

Source of every number: `outputs/metrics/*.csv`, produced by `notebooks/start.ipynb` (run of 2026-11-24). Environment: `outputs/setup.json` (Python 3.11.15, PyTorch 2.15.0, Apple M2, macOS 26.3.1, INT8 engine `qnnpack`, 1 benchmark thread). Seeds: 42, 123, 2026. Values are mean ± sample SD across seeds. "pp" means percentage points. Every change is paired, which means each variant is compared with the FP32 baseline trained with the same seed.

## 1. Baseline

| Seed | Chosen epoch (lowest validation loss) | Validation loss | Validation accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|---|---|
| 42 | 9 | 0.3058 | 0.8918 | 0.8858 | 0.8849 |
| 123 | 8 | 0.3032 | 0.8867 | 0.8795 | 0.8786 |
| 2026 | 6 | 0.3044 | 0.8928 | 0.8805 | 0.8797 |
| **Mean** | | | | **0.8819 ± 0.0034** | **0.8811** |

Validation loss stops improving after epochs 6–9 while training loss keeps falling (`outputs/figures/learning-curves.png`). That is why the checkpoint is chosen by validation loss rather than taken from the last epoch. Shirt is the hardest class (mean F1 0.683), and it is mostly confused with T-shirt/top, Pullover and Coat (`confusion-fp32.png`).

## 2. Main comparison

| Variant | Measured sparsity | Accuracy | Change (pp) | Macro F1 | Saved bytes | Bits / parameter | Size ratio | p50 ms, batch 1 | p50 ms, batch 64 |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 0.000 | 0.8819 ± 0.0034 | 0.00 | 0.8811 | 943,357 | 32.09 | 1.00× | 0.023 | 0.069 |
| pruned30 | 0.300 | 0.8821 ± 0.0040 | +0.01 ± 0.13 | 0.8813 | 943,357 | 32.09 | 1.00× | 0.023 | 0.069 |
| pruned50 | 0.500 | 0.8748 ± 0.0004 | −0.72 ± 0.36 | 0.8752 | 943,357 | 32.09 | 1.00× | 0.023 | 0.069 |
| pruned70 | 0.700 | 0.8182 ± 0.0171 | −6.38 ± 1.90 | 0.8219 | 943,357 | 32.09 | 1.00× | 0.023 | 0.069 |
| pruned90 | 0.900 | 0.4044 ± 0.0562 | −47.75 ± 5.84 | 0.3544 | 943,357 | 32.09 | 1.00× | 0.023 | 0.069 |
| int8 | 0.041 | 0.8822 ± 0.0028 | +0.03 ± 0.07 | 0.8813 | 240,987 | 8.20 | 3.91× | 0.114 | 0.239 |
| pruned50int8 | 0.500 | 0.8749 ± 0.0006 | −0.71 ± 0.38 | 0.8753 | 240,987 | 8.20 | 3.91× | 0.114 | 0.239 |

Bits per parameter is saved bytes × 8 ÷ 235,146 parameters, so it includes the overhead of PyTorch's file format. The p95 latencies are in `summary.csv`, and every individual timing is in `latency-raw.csv`.

**Decision rule** (fixed in advance in `protocol.md`): the smallest artifact whose mean accuracy is within 1 pp of FP32. The winner is **int8**: 3.91× smaller with +0.03 pp accuracy. pruned30, pruned50 and pruned50int8 also meet the accuracy limit. Dense pruning gives no storage advantage, and pruned50int8 is the same size as int8 but 0.7 pp less accurate.

Figures: `quality-size.png`, `quality-latency.png`, `pruning-curve.png`.

## 3. Findings

1. **Zeros are not compression.** Measured sparsity equals the target, but every pruned file is exactly as large as FP32, because a dense state dictionary stores each zero as a 32-bit float.
2. **Zeros are not speed either.** All pruned variants run within a few percent of FP32 latency, which is inside timing noise. Dense matrix multiplication does the same work whether or not the weights are zero.
3. **INT8 is the only method that reduced storage,** and it did so without losing accuracy. Per-class F1 changes by less than 0.2 pp. Only about 9 test images per seed switch from right to wrong, while about 12 switch from wrong to right (`comparisons.csv`).
4. **INT8 was slower on this hardware:** 5.0× at batch 1 and 3.5× at batch 64. Dynamic quantization converts the activations to INT8 on every call. For layers this small, that overhead costs more than the faster INT8 arithmetic saves. This result is specific to `qnnpack` on Apple M2 and should not be generalised. x86/fbgemm was not tested.
5. **Pruning has a steep drop between 70% and 90%:** accuracy goes 0.882 → 0.875 → 0.818 → 0.404 at 30/50/70/90%. The 90% result varies most from seed to seed (0.356, 0.391, 0.466). A single run at 90% therefore isn't a reliable estimate. The pilot's 0.4655 lies inside this range.
6. **At 90% the model falls back to a few classes.** Sandal and Shirt make up 54–71% of all predictions, against 20% of the true labels. Mean recall is 0.000 for Ankle boot and 0.010 for Coat (`prediction-collapse.png`, `confusion-pruned90.png`).
7. **Global pruning spreads sparsity unevenly.** At 90%, the hidden layers lose about 90% of their weights and the output layer about 70%. About 30 of the 256 first-layer neurons lose every incoming weight at 70% and 90% pruning, and 13 second-layer neurons also go dead at 90% (`layer-sparsity.png`, `layersparsity.csv`). The output layer has proportionally more large weights (`weight-distribution.png`), so one shared cut-off removes fewer of them.
8. **Coat and Shirt degrade first.** At 70% pruning, Coat F1 drops 21.0 pp and Shirt F1 drops 10.5 pp. The other classes lose between 0.9 pp (Trouser) and 6.7 pp (Pullover). These are the classes the baseline already confuses most.
9. **INT8 rounding makes 4.1% of weights exactly zero** with no pruning. These are weights smaller than half of one quantization step.

Example-level evidence: `failures-seed42.csv` lists every test image (by ID) that FP32 classifies correctly and a variant gets wrong. `failures-pruned70.png` shows the first 12 such images in test-index order; they were not hand-picked.

## 4. Limitations

- **Three seeds** is the protocol minimum, so the SDs are rough estimates. Five seeds remains the target (add 2027 and 2028 to `src/config.py`).
- **One model, one dataset, one machine.** The latency findings apply only to Apple M2 with `qnnpack`.
- **Timing noise.** Each forward pass takes well under 1 ms, so the differences between pruned variants are within noise. Every raw timing is saved so this can be checked.
- **Dense storage only.** A sparse format such as CSR could shrink the pruned files, but its size would have to include the index overhead. Faster inference would also need a real sparse kernel, which was not tested.
- **One-shot pruning without fine-tuning**, by design, to isolate the effect of pruning alone. Fine-tuning would probably recover much of the 70–90% loss, but it would be a separate experiment.
- **API status.** `torch.ao.quantization` is deprecated in PyTorch 2.15, and its replacement is `torchao`. It still works and gives valid INT8 models here.
- **Test-set exposure.** The seed-42 test set was seen during the pilot (`progress.md`, October 2026). The protocol was frozen before the repeated runs, and no setting was tuned on test results.

## 5. Comparison with the pilot runs

| Run | Seeds | Checkpoint | Baseline accuracy | 70% pruning | 90% pruning | INT8 bytes |
|---|---|---|---|---|---|---|
| Remote-server pilot (2026-10-30) | 42 | lowest validation loss | 0.8802 | 0.7283 | not run | 241,090 |
| Laptop pilot notebook (`start (1).ipynb`) | 42 | last epoch | 0.8825 | 0.8349 | 0.4655 | 241,469 |
| **Final (this document)** | 42, 123, 2026 | lowest validation loss | 0.8819 ± 0.0034 | 0.8182 ± 0.0171 | 0.4044 ± 0.0562 | 240,987 |

The pilots used different hardware, software and checkpoint rules. The large differences at 70% and 90% between them are the seed and run sensitivity described in finding 5. The pilots are exploratory, and only the final row is used for conclusions. File sizes differ by a few hundred bytes between runs because the PyTorch file format stores the file name and the layer names inside the file.
