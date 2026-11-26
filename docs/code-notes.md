# Code notes

This document explains the current code: each `src/` module, then each notebook section. Read it next to the code. It replaces the October 2026 snippet-by-snippet record for the remote pilot.

## Terms used below

- **Seed:** a starting number for a random-number generator. The same seed always gives the same "random" sequence, so shuffles, splits and initial weights repeat exactly. Different seeds give different but equally valid runs. Comparing seeds shows how much a result depends on luck.
- **Paired comparison:** every compressed variant is made from the baseline trained with the same seed, so differences come from the compression and not from a different training run.
- **State dictionary:** PyTorch's mapping from parameter names to tensors. Saving only this (no optimizer, no history) gives the inference-only artifact whose size we report.
- **Sparsity:** the fraction of weights that are exactly zero.
- **p50, p95, IQR:** the median, the 95th-percentile timing, and the spread between the 25th and 75th percentiles. Medians resist occasional slow outliers; p95 shows how bad the slow runs are.

---

## `src/config.py`: settings

Every value that defines the experiment lives here, so the notebook never hard-codes a number.

- `root = Path(__file__).resolve().parents[1]` finds the project folder from the file's own location, so the code runs on any machine and from any working directory. The pilot used a hard-coded path from another computer, which is why this was changed.
- `seeds = [42, 123, 2026]` are the three paired runs. Adding `2027, 2028` reaches the five-seed target with no other change.
- `valfraction = 0.1` gives 6,000 validation images out of 60,000.
- `batchsize`, `learningrate` and `epochs` are the baseline training settings from `protocol.md`.
- `prunelevels = [0.3, 0.5, 0.7, 0.9]` and `combinedlevel = 0.5` set the pruning variants. 90% is the extension level (`decisions.md`).
- `tolerance = 1.0` is the pre-registered 1 percentage point accuracy limit.
- `benchthreads = 1`, `warmup = 30`, `repeats = 200` and `benchbatches = [1, 64]` are the benchmark settings.
- `settings()` returns all of these as a dictionary, and the notebook saves it to `outputs/runs/config.json` so results can be traced to the settings that made them.

## `src/data.py`: data and split

- `loaddatasets()` loads the official train and test partitions through `datasets.FashionMNIST`, with `download=True` so a fresh machine fetches the data into `data/`. `transforms.ToTensor()` converts each 28×28 image into a float tensor of shape `[1, 28, 28]` with values in [0, 1]. This is the only preprocessing, and every model gets the same input.
- `splitindices(count, seed)` creates a `torch.Generator`, seeds it, and uses `torch.randperm` to shuffle the numbers 0–59,999. The first 54,000 indices are training and the last 6,000 are validation. This is exactly what `random_split` does internally, so seed 42 reproduces the pilot's split, but here the indices are returned so they can be saved.
- `makeloaders(...)` wraps the two index lists in `Subset` and builds three `DataLoader`s. Only the training loader shuffles, using its own seeded generator, so the batch order also repeats. Validation and test loaders keep a fixed order. The function also returns the split so the notebook can save it as `split.json`.

## `src/model.py`: the MLP

- `nn.Flatten()` turns `[batch, 1, 28, 28]` into `[batch, 784]`.
- `nn.Sequential(Linear(784, 256), ReLU, Linear(256, 128), ReLU, Linear(128, 10))` is the fixed architecture. The layer names in saved files are `layers.0`, `layers.2` and `layers.4` (the ReLUs have no parameters).
- The output is 10 raw scores (logits). No softmax is applied because `CrossEntropyLoss` applies it internally.
- `countparams` adds up `numel()` over all parameters, giving 235,146: (784·256 + 256) + (256·128 + 128) + (128·10 + 10).

## `src/train.py`: baseline training

- `setseed(seed)` seeds Python's `random`, NumPy and PyTorch. It is called before `MLP()` so the initial weights depend only on the seed.
- `runepoch(model, loader, device, optimizer=None)` handles both training and validation:
  - If an optimizer is passed, it trains: `model.train(True)`, gradients enabled, `zero_grad → backward → step` for each batch.
  - Without an optimizer, it only measures. Gradients are off and the model is in eval mode.
  - The loss uses `reduction="sum"` so the epoch average is exact even when the last batch is smaller. The backward pass uses `loss / batch size`, which gives the same gradient as the usual mean loss.
  - It returns the average loss and accuracy.
- `trainbaseline(seed, ...)`:
  - Runs 10 epochs, recording train and validation loss and accuracy each epoch.
  - Whenever validation loss reaches a new minimum, it stores a `deepcopy` of the weights.
  - At the end it loads the best weights back into the model. This is the checkpoint rule from `decisions.md`, and the test set is never touched.
  - It returns the model on the CPU in eval mode, together with the history and the chosen epoch.

## `src/compress.py`: variants, sparsity and artifacts

- The `warnings.filterwarnings` lines hide four known PyTorch deprecation messages about the quantization API and storage classes. They do not affect results, and the reason is logged in `decisions.md`.
- `chooseengine()` sets `torch.backends.quantized.engine` to the first available of `x86`, `fbgemm` and `qnnpack`. On Apple Silicon only `qnnpack` exists. The engine is saved in `setup.json` because INT8 speed depends heavily on it.
- `prunemodel(model, amount)`:
  1. `copy.deepcopy` so the baseline is never changed.
  2. Collect `(layer, "weight")` for all three `Linear` layers; biases are excluded.
  3. `prune.global_unstructured(..., L1Unstructured, amount)` ranks all 234,752 weights together by absolute value and zeros the smallest `amount` fraction. Because it is global, a layer with many small weights loses more of them than a layer with large weights.
  4. `prune.remove` deletes PyTorch's temporary mask and keeps the zeros in the ordinary `weight` tensor, so the saved file has normal dense tensors.
- `quantizemodel(model)` calls `quantize_dynamic` on a CPU copy with `{nn.Linear}` and `torch.qint8`. Each `Linear` becomes a `DynamicQuantizedLinear` that stores INT8 weights with a scale and zero point, and converts activations to INT8 on the fly at each call.
- `makevariants(baseline)` builds all seven variants, each from the baseline (the combined one is pruned first, then quantized).
- `layerweights(model)` returns each layer's weights as floats. For quantized layers it calls `layer.weight().dequantize()`, so sparsity can be counted the same way for every variant.
- `sparsity(model)` counts exact zeros per layer, the overall fraction, and `deadunits`: output neurons whose whole weight row is zero, so they receive no input.
- `saveartifact(model, path)` saves `model.state_dict()` and returns the real file size from `os.path.getsize`. Every file is called `model.pt` because `torch.save` stores the file name inside the archive. Different names would change the size by a few bytes.
- `loadartifact(name, path)` rebuilds the matching structure (a plain `MLP`, or a quantized `MLP` for names ending in `int8`) and loads the saved weights. Reported metrics come from this reloaded copy.

## `src/evaluate.py`: test metrics

- `predict` runs the model under `torch.inference_mode()` and returns predicted and true labels in loader order (the test loader is not shuffled).
- `scores` gives accuracy and macro F1. Macro F1 averages the per-class F1 scores equally, so a collapse on one class is visible.
- `perclass` uses `precision_recall_fscore_support` with `zero_division=0`, which matters at 90% pruning, where some classes are never predicted.
- `confusion` returns the 10×10 count matrix (rows are true classes, columns are predicted classes).
- `comparepredictions` compares a variant with FP32 on the same images:
  - `newerrors`: FP32 was right and the variant is wrong.
  - `fixed`: the reverse.
  - `bothwrong`: both are wrong.
  - `disagree`: the two predict different classes.

## `src/benchmark.py`: CPU latency

- `measure(name, seed, model, images)`:
  - Fixes the thread count and moves the model to the CPU in eval mode.
  - For each batch size, slices the same real test images and runs 30 untimed warm-up passes, which settle caches and memory allocation.
  - Then times 200 passes, with only the forward call between the two `time.perf_counter()` readings.
  - Each timing is returned as its own row, so every statistic can be recomputed later.
- `summarize(times)` computes the 25th, 50th, 75th and 95th percentiles with NumPy and returns p50, p95 and IQR = p75 − p25.

## `src/plotresults.py`: figures

Each function takes saved tables or arrays and writes a PNG to `outputs/figures/` through `save()`. None of them runs a model.

- `learningcurves`: loss and accuracy per epoch for each seed. Training curves are dashed, validation curves are solid, and a dotted line marks the chosen epoch.
- `qualitysize`: one faint dot per seed, a large marker with an SD bar per variant, and a green band for the 1 pp tolerance. The y-axis starts at 0 so the 90% point is visible (the pilot's axis started at 0.7 and hid it).
- `qualitylatency`: one panel per batch size, with horizontal error bars showing the SD of p50 across seeds.
- `pruningcurve`: accuracy, macro F1 and measured sparsity against the pruning target, with 0% being FP32.
- `confusionplot`, `perclassf1`, `layersparsityplot` and `predictionhist` are the diagnostic charts.
- `weighthist`: per-layer |weight| histograms on a log scale, with each pruning cut-off drawn as a vertical line.
- `examplegrid`: test images with their ID, true label, FP32 prediction and variant prediction.

---

## `notebooks/start.ipynb`: section by section

1. **Environment.** Finds the project root (it works whether the notebook starts in `notebooks/` or the root), adds it to `sys.path` so `from src import ...` works, and creates `outputs/metrics` and `outputs/figures`. Records versions, OS, CPU name (`sysctl` on macOS), logical CPU count, training device, INT8 engine and benchmark threads to `setup.json`.
2. **Data.** Loads both partitions and checks counts (6,000 and 1,000 per class), shape `[1, 28, 28]` and pixel range [0, 1]. Writes `dataset-summary.json` and shows the first training image of each class.
3. **Model check.** One forward pass on 8 images to confirm the output shape `[8, 10]` and the 235,146 parameters.
4. **Protocol settings.** Saves `config.settings()` to `outputs/runs/config.json`.
5. **Training.** For each seed: builds loaders, asserts that train and validation indices do not overlap, saves `split.json`, trains, saves `history.csv` and prints the chosen epoch. Then draws the learning curves.
6. **Variants.** For each seed:
   - Builds the seven variants, then for each one saves `model.pt`, reloads it and evaluates the reloaded copy.
   - The `assert` checks that the reloaded predictions equal the in-memory ones, which proves the file is complete.
   - Records sparsity, scores, bytes and bits per parameter, plus per-class rows, per-layer rows, the confusion matrix and the comparison with FP32.
   - The next cell adds paired deltas (`deltaaccpp`, `deltaf1pp`) and the size ratio by looking up each row's own-seed FP32 values, then writes every table to `outputs/metrics/`.
7. **Benchmark.** Reloads every saved file and times it on the first 64 test images. Saves raw and summarised timings.
8. **Summary.** Groups by variant to get the mean, accuracy SD and delta SD, and adds the mean p50/p95 per batch size. Writes `summary.csv`, the main results table.
9. **Decision rule.** Keeps variants with `deltaaccpp >= -1`, sorts them by bytes, and reports the smallest.
10. **Trade-off figures.** The three required plots.
11. **Sparsity location.**
    - Computes each global cut-off as the k-th smallest |w| over all weights of the seed-42 baseline.
    - Checks that the 90% cut-off selects exactly the weights PyTorch zeroed (0 mismatches expected).
    - Plots the histograms and per-layer sparsity, and tabulates dead neurons.
12. **Failure analysis.**
    - Per-class F1 change against FP32, averaged over seeds, and the per-class F1 chart.
    - Example-change counts.
    - Every seed-42 image that FP32 classifies correctly and a variant gets wrong is saved to `failures-seed42.csv`. The grid shows the first 12 for `pruned70` in test-index order.
    - The FP32 confusion matrix.
13. **90% collapse.** Predicted-class counts for FP32, 70% and 90% (seed 42), the 90% confusion matrix, mean recall per class, and the per-seed 90% rows.
14. **Key findings** and 15. **Limitations.** Written from the saved summary. They match `docs/results.md`.
