# Progress log
This file is an evidence-based handoff log. Add newest entries at the top. Include date, tasks touched, exact files/commands where relevant, outcomes, and next action. A plan or attempted run is not a completed result.
## 2026-11-24 — Final three-seed study completed

- **Run:** `notebooks/start.ipynb`, executed top to bottom in a fresh `.venv` (`requirements.txt`) on Apple M2 / macOS 26.3.1 / Python 3.11.15 / PyTorch 2.15.0 / `qnnpack`. A full run takes about 3.3 minutes.
- **Evidence:** `outputs/setup.json`, `outputs/runs/`, `outputs/metrics/summary.csv` and the other CSVs, and 12 figures in `outputs/figures/`. A second complete run gave identical accuracy and byte values. Latency changed by a few hundredths of a millisecond.
- **Results (mean ± SD, 3 seeds):**
  - FP32: 0.8819 ± 0.0034, 943,357 bytes.
  - Pruning, change in accuracy: 30% +0.01 pp, 50% −0.72 pp, 70% −6.38 pp, 90% −47.75 ± 5.84 pp. All pruned files are 943,357 bytes.
  - INT8: +0.03 pp, 240,987 bytes (3.91× smaller), about 5.0× slower at batch 1 and 3.5× slower at batch 64 on this CPU.
  - 50% pruning + INT8: −0.71 pp, 240,987 bytes.
  - Decision rule: INT8 is selected.
- **Completed tasks:** T01, T10, T11, T20–T23, T30–T33, T40–T43, T50, T51 (see `tasks.md`).
- **Next:** T00 (teacher brief), then optionally seeds 2027 and 2028, then the report (T52) and presentation (T53).

## 2026-11-24 — Code consolidated into `src/` and one main notebook

- **Done:** Wrote `src/config.py`, `data.py`, `model.py`, `train.py`, `compress.py`, `evaluate.py`, `benchmark.py` and `plotresults.py`, and rebuilt `notebooks/start.ipynb` to use them.
  - Removed the unexecuted starter notebook.
  - Replaced the laptop pilot notebook; its results are recorded in the entry below.
  - Added `requirements.txt` and `.gitignore`.
- **Fixes over the pilot notebook:**
  - Project-relative paths instead of a hard-coded path from another machine.
  - Checkpoint chosen by validation loss.
  - Three seeds.
  - Every variant is saved under the same file name, reloaded, and checked before it is evaluated.
  - Benchmark follows the protocol: 1 thread, `inference_mode`, 30 warm-up and 200 timed passes, real images, raw timings saved.
  - Accuracy chart now starts at 0 (the pilot's axis started at 0.7 and hid the 90% bar).
  - Per-class, per-layer and example-level diagnostics added.
- **Docs:** Updated `README.md`, `agents.md`, `overview.md`, `architecture.md`, `protocol.md`, `tasks.md` and `decisions.md` to match the code. Added `results.md`. Rewrote `code-notes.md` for the current code.

## 2026-11-24 — 90% pruning result re-verified; code review

- **Request:** The teacher asked for the 90% pruning result to be re-verified.
- **Method:** An independent script reproduced the pilot notebook's setup with five seeds (42, 123, 2026, 2027, 2028). It also checked the pruning mask against a hand-computed global threshold and measured per-layer sparsity.
- **Findings:**
  - The pruning code is correct: the mask matches the hand-computed threshold exactly, and sparsity is exactly 90%.
  - The single-seed figure of 0.4655 is not stable. Five-seed accuracy was 0.397 ± 0.059, with a range of 0.347–0.467.
  - The 90% model predicts mostly Sandal and Shirt.
  - Global and per-layer 90% pruning were compared: per-layer was worse (0.319).
  - In a side check only, 2 epochs of fine-tuning with the zeros kept fixed recovered 0.838. This is not part of the study.
- **Review issues found in the pilot notebook:**
  - The accuracy chart axis started at 0.7 and hid the 90% bar.
  - The checkpoint was not chosen by validation loss.
  - Only one seed was run.
  - INT8 being slower than FP32 was left out of the key findings.
  - The latency method did not follow the protocol.
  - Pruned files were 12 bytes larger because of the longer file name.
  - The path was hard-coded to another machine.
  - The README links were broken.
- **Next:** Fix these issues in a consolidated pipeline (entry above).

## 2026-11-24 — Laptop pilot notebook received

- **Observed evidence:** `start (1).ipynb`, which ran on a different Mac (Python 3.12.4, PyTorch 2.15.0, CPU, `qnnpack`). The run date is not recorded in the file. Its seed-42 test results with the last-epoch checkpoint:
  - Baseline 0.8825.
  - Pruning: 30% 0.8825, 50% 0.8837, 70% 0.8349, 90% 0.4655. All pruned files were 943,725 bytes, against 943,713 for the baseline.
  - INT8: 0.8818, 241,469 bytes, p50 latency 0.29 ms against 0.03 ms for FP32 at batch 1.
- **Status:** Exploratory pilot. The notebook file itself was later removed from the project folder; this entry is its record.

## 2026-10-31 — Final-study scope clarified from the related-work note
- **Documentation change only:** Refined `overview.md`, `protocol.md`, `tasks.md`, and `decisions.md` to distinguish the seed-42 pilot from the final study.
- **Planned final scope:** Target five paired seeds with three as the minimum; compare using actual serialized bytes and a predeclared one-percentage-point accuracy tolerance; report CPU p50/p95/IQR; add layer-wise weight-distribution and class/example failure analyses.
- **Boundary:** Sparse-encoded storage, if attempted, is distinct from dense checkpoint size and cannot imply faster inference without a measured sparse runtime. Structured pruning, QAT, INT4, ONNX serving, and CIFAR/ResNet remain optional future extensions.
- **No new experiment was run:** The previously reported pilot scores are unchanged. First-lab sparsity/latency/plot snippets still need execution in the remote notebook.
## 2026-10-30 — First comparison table generated
- **Observed evidence:** User screenshot shows a five-model comparison generated from saved metrics and checkpoint files.
- **Results:** Baseline: `0.8802` accuracy, `0.8808375` macro F1, `943314` bytes. Pruned 30%: `0.8782`, `0.8793731`, `943324` bytes. Pruned 50%: `0.8636`, `0.8664769`, `943324` bytes. Pruned 70%: `0.7283`, `0.7378090`, `943324` bytes. Dynamic INT8: `0.8811`, `0.8816217`, `241090` bytes.
- **Interpretation:** The unstructured-pruned checkpoints are effectively the same size as baseline (10 extra bytes of serialization overhead); zeros alone did not reduce storage. Dynamic INT8 is about 74.44% smaller than baseline with near-baseline test metrics. Its slightly higher score is one run and is not evidence of a robust gain.
- **Completed:** Metrics-and-size comparison table ran and its output was shared. Updated the task handoff with these measured sizes.
- **Not done:** Actual pruning sparsity values, CPU latency benchmarks, repeated-seed runs, compressed-model comparison plots, and final report/presentation remain.
- **Next:** Capture measured sparsity and run consistent CPU latency benchmarks for all variants; then do repeated seeds and final plots/analysis.
## 2026-10-30 — Dynamic INT8 test evaluation saved; comparison table prepared
- **Observed evidence:** User screenshot shows 10,000 INT8 test examples, accuracy `0.8811`, macro F1 `0.8816216815598622`, and the `Saved INT8 test results.` confirmation.
- **Comparison:** Against baseline accuracy `0.8802` and macro F1 `0.8808375164504273`, INT8 is higher by 0.09 and about 0.078 percentage points respectively in this run; this small difference should not be framed as a robust improvement without repeated seeds.
- **Completed:** Added a standard-library helper and notebook call to build a comparison CSV from saved metrics and actual checkpoint byte sizes.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` sections 52–53.
- **Not done:** Artifact sizes for the variants (except the previously shared 50% size) have not been shared; the generated comparison CSV has not been confirmed run.
- **Next:** Run sections 52–53 once all five metric and checkpoint paths exist, then review the resulting table.
## 2026-10-30 — Dynamic INT8 artifact reload and inference succeeded
- **Observed evidence:** User screenshot shows reloaded output shape `[128, 10]` and `All scores finite: True` after loading the saved dynamic INT8 state dictionary into a fresh quantized model.
- **Completed:** The INT8 artifact reload and one-batch CPU inference check succeeded.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 50.
- **Warning:** PyTorch emitted a `TypedStorage` deprecation warning. It did not interrupt loading or inference.
- **Not done:** Full test metrics have not been shared; quantized artifact path/size are not shown in this screenshot.
- **Next:** Run section 51 to evaluate all 10,000 test examples and save metrics under `dynamic-int8`.
## 2026-10-30 — Quantization engine confirmed; INT8 helper prepared
- **Observed evidence:** User screenshot shows PyTorch `2.4.0a0+f70bd71a48.nv24.06`; supported engines are `qnnpack`, `none`, `onednn`, `x86`, `fbgemm`; current engine is `x86`.
- **Completed:** Added a reusable dynamic INT8 helper that copies the float model to CPU and converts `Linear` layers with the reported `x86` engine.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 47.
- **Not done:** The helper has not been confirmed saved or run remotely; no quantized artifact or metrics exist yet.
- **Next:** Add and save the helper in `/workspace/ml/src/compress.py`, then create and smoke-test the CPU quantized model.
## 2026-10-30 — 70% pruning evaluation saved; quantization backend check prepared
- **Observed evidence:** The user shared the 70% pruned model's test results for 10,000 examples: accuracy `0.7283`, macro F1 `0.7378089777219896`, and save confirmation. Compared with the baseline (`0.8802`, `0.8808375164504273`), the decrease is 15.19 accuracy points and about 14.30 macro-F1 points.
- **Completed:** Added the result to the handoff and prepared a notebook cell to inspect PyTorch's supported and current quantized engines.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 46.
- **Not done:** The quantized-engine output and 70% measured sparsity/artifact byte size were not included in the screenshot. No quantized model has been created.
- **Next:** Run section 46 and use the reported supported-engine list to choose the dynamic INT8 backend.
## 2026-10-30 — 50% pruned artifact saved and reloaded
- **Observed evidence:** The user shared artifact path `/workspace/ml/outputs/checkpoints/pruned-50.pt`, file size `943324` bytes, and reloaded output shape `[128, 10]`.
- **Completed:** Serialization and a one-batch forward pass for the 50% variant succeeded; added its test-evaluation and result-saving notebook cell.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 42.
- **Not done:** The 50% test metrics and CSV outputs are not yet confirmed. Exact 50% measured sparsity is not included in the screenshot.
- **Next:** Evaluate `checkmodel50` on the official test set, save its outputs, and record the resulting metrics.
## 2026-10-30 — 50% pruning evaluation saved; 70% variant prepared
- **Observed evidence:** The user shared 50% pruning results for 10,000 test examples: accuracy `0.8636`, macro F1 `0.8664769072067122`, and save confirmation. Relative to baseline (`0.8802`, `0.8808375164504273`), changes are -1.66 and about -1.44 percentage points.
- **Completed:** Added a cell to create the 70% global L1 variant independently from the baseline and measure actual sparsity.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 43.
- **Not done:** The 70% artifact and test results are pending; measured sparsities and file sizes for prior variants have not been shared.
- **Next:** Create `model70` from the baseline, then save/reload and evaluate it with its own output paths.
## 2026-10-30 — Added 50% artifact save and reload cell
- **Completed:** Added a notebook cell that refreshes `compress.py`, saves the 50% pruned state dictionary, measures file bytes, reloads it into a fresh model, and checks one batch output shape.
- **Evidence:** `docs/code-notes.md` section 41 and `docs/tasks.md` current handoff.
- **Not done:** The 50% variant, artifact, size, and reload check have not been confirmed remotely.
- **Next:** Run the 50% creation cell followed by this save/reload cell; use `checkmodel50` for evaluation only if the output shape is correct.
## 2026-10-30 — 30% pruning evaluation saved; 50% variant prepared
- **Observed evidence:** The user shared the 30% pruned model's test results for 10,000 examples: accuracy `0.8782`, macro F1 `0.8793730888992333`; the notebook printed `Saved 30% pruning test results.` Compared with baseline, changes are -0.20 and about -0.146 percentage points, respectively.
- **Completed:** Added a cell to create a 50% global L1-pruned copy from the original baseline model and report measured sparsity.
- **Evidence:** User-provided screenshot; `docs/code-notes.md` section 40.
- **Not done:** The 50% artifact has not been created/saved/evaluated. Exact 30% measured sparsity and artifact size were not provided.
- **Next:** Run section 40 with `model` still holding the baseline checkpoint; then save, reload, and evaluate the 50% variant in its own output paths.
## 2026-10-30 — Added 30% pruning evaluation cell
- **Completed:** Added a notebook cell to evaluate the reloaded 30% pruned artifact with the same test evaluator as the baseline and save its results to a distinct output directory.
- **Evidence:** `docs/code-notes.md` section 39 and `docs/tasks.md` current handoff.
- **Not done:** Pruned artifact save/reload and its test evaluation have not been confirmed remotely; no pruning metrics are known.
- **Next:** Run the artifact cell, confirm a `[batch size, 10]` output, then run the 30% evaluation cell.
## 2026-10-30 — Combined compression-module reload and artifact check
- **Completed:** Added a notebook cell that reloads `compress`, checks for `saveartifact`, and if available saves and reloads the 30% pruned model for a forward-pass check.
- **Evidence:** `docs/code-notes.md` section 38 and `docs/tasks.md` current handoff.
- **Not done:** The remote save helper and pruned artifact remain unconfirmed; the new cell has not been run.
- **Next:** Run the cell with `model30`, `device`, and `testloader` available; confirm the helper exists and output shape is `[batch size, 10]`.
## 2026-10-30 — Diagnosed pruned-artifact helper import failure
- **Observed evidence:** The user shared an `ImportError` for `saveartifact` from `/workspace/ml/src/compress.py`. The traceback points to the import line, before the save helper was called.

- **Likely cause:** The notebook may have cached `compress` after importing `prunemodel`, before `saveartifact` was added; alternatively, the helper may not have been saved in that file.
- **Completed:** Added a module reload and function-availability check.
- **Evidence:** `docs/code-notes.md` section 37 and `docs/tasks.md` current handoff.
- **Not done:** The pruned artifact has not been confirmed saved; no file size or reload output is available.
- **Next:** Reload `compress`, confirm `saveartifact` is present, then save and smoke-check the artifact.
## 2026-10-30 — Added pruning artifact save and reload snippets
- **Completed:** Added a helper to save a model state dictionary and report exact file bytes, plus a notebook cell to reload the 30% pruned model and check an inference output shape.
- **Evidence:** `docs/code-notes.md` sections 35–36 and `docs/tasks.md` current handoff.
- **Not done:** The pruning/artifact code has not been confirmed saved or run remotely; no pruned artifact size or reload output is known.
- **Next:** Create the 30% variant, save it, and confirm the reloaded output shape is `[batch size, 10]`.
## 2026-10-30 — Added first pruning variant snippets
- **Completed:** Added a reusable global L1 pruning helper that copies the baseline, prunes linear weights, makes zeros permanent, and returns measured sparsity; added a notebook cell to create a 30% variant.
- **Evidence:** `docs/code-notes.md` sections 33–34 and `docs/tasks.md` current handoff.
- **Not done:** The pruning code has not been saved or run remotely; no pruned model artifact or test result exists yet.
- **Next:** Confirm baseline outputs as needed, save `compress.py`, create the 30% copy from the validation-selected baseline, and report its measured sparsity.
## 2026-10-30 — Added confusion-matrix plotting snippets
- **Completed:** Added a `plotconfusion` function to the code notes and a notebook call to save/display the labeled confusion-matrix heatmap.
- **Evidence:** `docs/code-notes.md` sections 31–32 and `docs/tasks.md` current handoff.
- **Not done:** The CSV save step and both figures are not confirmed run remotely.
- **Next:** Confirm the CSV artifacts exist, save the updated plotting module, and run the confusion-matrix plotting cell.
## 2026-10-30 — Added baseline learning-curve plotting snippets
- **Completed:** Added a focused `src/plotresults.py` helper to plot saved training/validation loss and accuracy, plus a notebook cell to save and display the figure.
- **Evidence:** `docs/code-notes.md` sections 29–30 and `docs/tasks.md` current handoff.
- **Not done:** CSV artifact saving has not been confirmed; the plotting module and figure have not been created remotely.
- **Next:** Confirm the evaluation CSV files exist, save `plotresults.py`, and run the plotting cell.
## 2026-10-30 — Added reload, save, and artifact verification cells
- **Completed:** Added notebook steps that reload `evaluate.py`, save the existing baseline test results when the helper is available, and verify all three CSV files.
- **Evidence:** `docs/code-notes.md` sections 27–28 and `docs/tasks.md` current handoff.
- **Not done:** The remote save function and output files are still unconfirmed; the notebook cells have not been run.
- **Next:** Run the reload/save cell, resolve the missing-function message if shown, and confirm all three files exist.
## 2026-10-30 — Diagnosed evaluation save-function import failure
- **Observed evidence:** The user shared an `ImportError` for `saveoutputs` from `/workspace/ml/src/evaluate.py`. The traceback points to the import line, before the save function ran.
- **Likely cause:** The notebook may have cached `evaluate` before `saveoutputs` was added; alternatively, the function may not have been saved in the remote file.
- **Completed:** Added a module reload and function-availability check, with next steps for either outcome.
- **Evidence:** `docs/code-notes.md` section 26 and `docs/tasks.md` current handoff.
- **Not done:** No CSV artifacts are confirmed written. The test evaluation does not need to be repeated.
- **Next:** Reload `evaluate`, confirm `saveoutputs` is available, and rerun the save call.
## 2026-10-30 — Official baseline test evaluation completed
- **Observed evidence:** The user shared test metrics for all 10,000 Fashion-MNIST test examples: accuracy `0.8802`, macro F1 `0.8808375164504273`. Per-class F1 ranged from Shirt `0.690188...` (lowest) to Trouser `0.979899...` (highest). These are the first reported official test results for the uncompressed baseline.
- **Completed:** Added functions and notebook instructions to save per-example predictions, per-class metrics, and the confusion matrix as CSV files.
- **Evidence:** User-provided test-output screenshot; `docs/code-notes.md` sections 24–25.
- **Not done:** The CSV artifacts have not been confirmed written; no compression variants have been evaluated.
- **Next:** Run the save cell, verify the three CSV outputs, then create the baseline learning curves and confusion-matrix figure before beginning compression experiments.
## 2026-10-30 — Full baseline training completed; test evaluation prepared
- **Observed evidence:** The user shared output for all 10 epochs. Training loss decreased from about `0.5903` to `0.2340`, and training accuracy rose from about `0.7906` to `0.9129`. Validation loss reached its minimum at epoch 8 (`0.3070`) and then rose to `0.3188` by epoch 10; validation accuracy was `0.8832` at epoch 10. This run used the configured checkpoint rule, which selects the lowest validation loss.
- **Completed:** Added evaluation module and notebook snippets for collecting predictions, calculating accuracy/macro F1/per-class metrics/confusion matrix, and loading the selected checkpoint for official test evaluation.
- **Evidence:** `docs/code-notes.md` sections 21–23; user-provided 10-epoch screenshot.
- **Not done:** The checkpoint and history CSV are not confirmed present; test evaluation has not been run, and no test result is reported here.

- **Next:** Run the checkpoint load and test evaluation once, then save predictions and confusion-matrix artifacts.
## 2026-10-30 — Diagnosed baseline function import failure
- **Observed evidence:** The user shared an `ImportError` for `trainbaseline` from `/workspace/ml/src/train.py`. The traceback points to the import line, so the full training function was not called.
- **Likely cause:** The notebook may have cached the earlier version of `train` before `trainbaseline` was added; alternatively, the updated function may not have been saved in that file.
- **Completed:** Added a notebook module-reload and availability check, with instructions for both outcomes.
- **Evidence:** `docs/code-notes.md` section 20 and `docs/tasks.md` current handoff.
- **Next:** Reload `train`, confirm `trainbaseline` is available, then rerun the import and baseline cell.
## 2026-10-30 — Smoke training completed; full baseline snippets prepared
- **Observed evidence:** The user shared a notebook result from one training pass: training loss `0.5815139833114765`, training accuracy `0.7948148148148149`, validation loss `0.4295896174112956`, and validation accuracy `0.853`. These values are recorded as a one-epoch smoke run only.
- **Completed:** Added code-note snippets to reset the model after the smoke run, train for up to 10 epochs, select the checkpoint by lowest validation loss, and save per-epoch metrics to CSV. Updated the protocol and decision log with the checkpoint criterion.
- **Evidence:** `docs/code-notes.md` sections 17–19, `docs/protocol.md`, and `docs/decisions.md`.
- **Not done:** The full training function has not been confirmed saved or run remotely; no final baseline checkpoint or history file is confirmed.
- **Next:** Save the helper in `/workspace/ml/src/train.py`, rerun the model reset cell, and run the full baseline cell.
## 2026-10-30 — Added notebook baseline initialization and smoke run
- **Completed:** Added notebook snippets to initialize the model, loss function, optimizer, and device, then run one training and validation pass. Recorded explanations in the code notes and updated the task handoff.
- **Evidence:** `docs/code-notes.md` sections 15–16 and `docs/tasks.md` current handoff.
- **Not done:** The snippets are not confirmed saved or executed on the remote server; no training metrics or baseline checkpoint are confirmed.
- **Next:** Rerun the notebook setup/data cells after any kernel restart, verify model import, then run these cells. If the smoke run works, add the full epoch loop with history and checkpoint selection.
## 2026-10-30 — Added training and validation helpers
- **Completed:** Added device selection, one-epoch training, and validation snippets for `src/train.py`, with detailed line explanations.
- **Evidence:** `docs/code-notes.md` sections 12–14.
- **Not done:** These are code snippets provided for the remote workspace; they are not confirmed saved or executed. No training has been confirmed.
- **Next:** Restore notebook variables after the kernel restart, verify the model output, save `config.py` and `train.py`, then run one training and validation pass from a notebook cell.
## 2026-10-30 — Added training seed helper snippet
- **Completed:** Added a small `setseed` helper for `src/train.py` and its detailed explanation to `docs/code-notes.md`.
- **Evidence:** `docs/code-notes.md` section 11.
- **Not done:** The user has not confirmed saving the config or running the model check; no training was performed.
- **Next:** Restore variables after the kernel restart, verify the model check, save `config.py`, and continue with the training function.
## 2026-10-30 — Added baseline settings snippet
- **Completed:** Added the small `src/config.py` baseline settings snippet and line-by-line explanation to `docs/code-notes.md`; updated the task handoff.
- **Evidence:** `docs/code-notes.md` section 10 and `docs/tasks.md` current handoff.
- **Not done:** The user's model check showed `NameError: device is not defined` after a kernel restart. The settings snippet has been provided but is not confirmed saved. No training has happened.
- **Next:** Rerun setup, data, split, and loader cells after the kernel restart; confirm the model smoke check; then save the settings module before training.
## 2026-10-30 — Added model smoke-check snippet
- **Completed:** Added the notebook code to import `fashionmlp`, run one batch, and report output shape and parameter count to `docs/code-notes.md`, with line-by-line explanations. Updated the task handoff.
- **Evidence:** `docs/code-notes.md` section 9 and `docs/tasks.md` current handoff.
- **Not done:** The snippet has been provided but has not been confirmed as run; no training was done.
- **Next:** Run the smoke check in `/workspace/ml/notebooks/start.ipynb` and compare the output shape and parameter count with the documented expectations.
## 2026-10-30 — Added a code and explanation record
- **Completed:** Created `docs/code-notes.md` with the active setup, data, split, loader, audit, and model snippets and detailed line explanations. Added the rule in `agents.md` to append future snippets and explanations, and linked the record from the README. Updated the task handoff to point to the next model check.
- **Evidence:** `docs/code-notes.md`, `agents.md`, `README.md`, and `docs/tasks.md`.
- **Not done:** No remote code was executed or changed by this documentation update; no training or model evaluation has happened.
- **Next:** Confirm the split/loaders and `src/model.py` exist on the remote server, then check model output shape and parameter count.
## 2026-10-30 — Workspace skeleton and starter notebook created
- **Completed:** Created the project directories (`data/`, `notebooks/`, `outputs/`) and implemented `notebooks/start.ipynb` containing the 5 structured cells for environment verification, quantization engine inspection, Fashion-MNIST loading, class distribution inspection, sample grid visualization, and deterministic 90/10 train-validation splitting.
- **Evidence:** `data/`, `notebooks/start.ipynb`, `outputs/`.
- **Not done:** Remote execution of the notebook on `/workspace/ml` has not yet taken place; model definition and training (Phase 2) are pending.
- **Next:** Run the starter notebook cells on the remote Jupyter server, record `outputs/setup.json` and `outputs/dataset-summary.json`, then implement the baseline MLP model in `src/model.py`.
## 2026-10-29 — README shortened and agent guidance centralized
- **Completed:** Reduced the README to a short project description and document links. Added root `agents.md` with project context, naming and code-snippet preferences, workspace guidance, experiment integrity rules, and the reported remote environment. Removed repeated coding-style guidance from the architecture, protocol, decisions, and task documents.
- **Evidence:** `README.md`, `agents.md`, and the updated files under `docs/`.
- **Not done:** No remote folders, notebook, dataset download, model code, or experiment results were created.
- **Next:** Create the remote workspace and starter notebook when ready to begin implementation.
## 2026-10-29 — Workspace layout and naming guidance updated
- **Completed:** Updated the project documents to use lowercase document names and the agreed remote layout: `/workspace/ml/data`, `/workspace/ml/notebooks`, and `/workspace/ml/outputs`. Documented `notebooks/start.ipynb`, `outputs/setup.json`, and the rule to add `src/` when reusable code emerges.
- **Evidence:** `README.md` and the files under `docs/`.
- **Not done:** No remote folders or notebook were created by this documentation change; no dataset was downloaded and no model was trained.
- **Next:** Create the three initial folders and `notebooks/start.ipynb` on the remote server, then run the environment and dataset check.
## 2026-10-29 — Documentation scaffold
- **Completed:** Created project overview, architecture plan, experiment protocol, detailed task backlog, decision log, and reference list.
- **Evidence:** `README.md` and files under `docs/`.
- **Observed workspace state:** Only the prior fall-detection data folders (`KFall_dataset/`, `SisFall_dataset/`) were present; no compression code, Fashion-MNIST cache, checkpoints, or result tables were present.
- **Not done:** No data was downloaded, no code was implemented, and no model was trained or benchmarked.
- **Next:** Complete T00 (confirm teacher requirements), then T01/T10 (environment and data setup).
