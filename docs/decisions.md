# Decision log

Record decisions that affect project scope or experimental validity. Add newest entries at the top. Each entry states the choice, the reason, alternatives considered, and consequences.

## Open decisions

- Confirm the teacher's brief, required report format and deadline (T00).
- Decide whether to run seeds 2027 and 2028 before submission (five-seed target).
- Decide whether to add an extension (fine-tuning after pruning, structured pruning, x86 latency), and only with the teacher's agreement.

## 2026-11-24 — Consolidate on one notebook plus `src/` modules, run locally

- **Choice:** The laptop notebook becomes the single main notebook, `notebooks/start.ipynb`. Its logic is moved into the `src/` modules described in `architecture.md`, and all paths are relative to the project folder. The unexecuted starter notebook was removed. The laptop pilot notebook (`start (1).ipynb`) was replaced; its results are recorded in `progress.md` and `results.md`.
- **Reason:** The pilot notebook used a hard-coded path from another machine and could not run elsewhere. The remote `/workspace/ml` code only existed as snippets in the old code notes. One reproducible entry point is needed for marking.
- **Alternatives:** Keep everything in one notebook. This was rejected because repeated seeds reuse the same code, and `agents.md` asks for `src/` once code is reused.
- **Consequence:** `requirements.txt` pins the versions used. The remote server is no longer required, and its October results stay as a pilot in `progress.md`.

## 2026-11-24 — Restore the validation-loss checkpoint rule

- **Choice:** Use the lowest-validation-loss epoch, as decided on 2026-10-30.
- **Reason:** The laptop pilot notebook used the last epoch and never computed validation loss, which did not follow the protocol.
- **Consequence:** The chosen epochs are 9, 8 and 6 for seeds 42, 123 and 2026.

## 2026-11-24 — Add 90% pruning as an extension level

- **Choice:** Add 90% to the pruning levels (30/50/70 remain the core).
- **Reason:** The laptop pilot already included 90%, and the teacher asked for it to be re-verified. It locates where accuracy collapses. This level was added after the pilot had been seen, which is disclosed here and in `results.md`. It does not affect the decision rule, because 90% is far outside the 1 pp tolerance.
- **Verification:** A hand-computed global cut-off produces exactly the same mask as PyTorch's pruning (shown in notebook Section 11). Measured sparsity is 90.00%.

## 2026-11-24 — Include the combined 50% prune → INT8 variant

- **Choice:** Run the optional combined variant for every seed.
- **Reason:** Saving, reloading, prediction and timing all work under the same runtime, which was the condition set in the protocol.

## 2026-11-24 — Three paired seeds for the final study

- **Choice:** Seeds 42, 123 and 2026, the protocol minimum. Five remains the target.
- **Reason:** Three seeds meet the minimum standard and show the seed sensitivity of the 70% and 90% results. Two more seeds would take about two extra minutes to run.
- **Consequence:** Results report mean ± sample SD. No significance tests are claimed with n = 3.

## 2026-11-24 — Fix the latency benchmark and artifact naming

- **Choice:**
  - 1 CPU thread, `torch.inference_mode()`, 30 warm-up and 200 timed passes.
  - Real test images as input, batch sizes 1 and 64, every timing saved.
  - Every model is timed and evaluated after reloading its saved file.
  - Every artifact is named `model.pt`.
- **Reason:** The pilot used 10 warm-ups, random-noise input, the default thread count, and no saved raw timings. It also named files differently, which changed their byte counts (pruned files were 12 bytes larger only because of the longer name).
- **Alternatives:** Using all cores. Rejected because single-thread timing is more stable on a laptop, and the thread count is recorded.

## 2026-11-24 — INT8 engine choice and deprecated API

- **Choice:** Use the first available engine among `x86`, `fbgemm` and `qnnpack`, and record it. Keep `torch.ao.quantization.quantize_dynamic`.
- **Reason:** `qnnpack` is the only engine on Apple Silicon. `torch.ao.quantization` is deprecated in PyTorch 2.15 but still produces valid INT8 models; moving to `torchao` would change the method in the middle of the study.
- **Consequence:** The latency results apply to `qnnpack` on Apple M2 only. This is listed as a limitation.

## 2026-10-31 — Refine the comparison around real budgets and repeatability

- **Decision:** Treat seed-42 outputs as a pilot. The final study targets five paired seeds and requires at least three; compare variants by measured artifact bytes under a predeclared 1 percentage-point mean-accuracy tolerance, not by nominal pruning percentage alone. Include p50, p95, and IQR CPU latency. Add layer-wise weight-distribution and class/example failure diagnostics after the core runs.
- **Reason:** The project should directly address the pasted literature review's concern that a table of one pruning run and one quantized run is too weak. Actual storage representation, repeatability, and failure analysis make the comparison more informative.
- **Scope boundary:** Keep one MLP, Fashion-MNIST, global magnitude pruning at 30/50/70%, and dynamic INT8 as the core. Sparse encoding is a separately labeled representation and is only a storage result unless a real sparse inference path is benchmarked. Structured pruning, QAT/static PTQ, INT4, ONNX serving, and CIFAR/ResNet remain out of scope unless the teacher asks for an extension.
- **Consequence:** Update `overview.md`, `protocol.md`, and `tasks.md`. Existing scores stay pilot results; no new sparsity or latency values are inferred from this plan.

## 2026-10-30 — Select the baseline checkpoint by validation loss

- **Decision:** Save the epoch checkpoint with the lowest validation loss; report validation accuracy alongside it.
- **Reason:** Validation loss uses the model's confidence as well as the predicted class, making it a useful checkpoint-selection signal. The official test set remains unused for selection.
- **Scope:** Initial baseline protocol; this does not change the architecture, optimizer, data split, or test procedure.


## 2026-10-29 — Keep the initial workspace small and modular

- **Choice:** Use `/workspace/ml` with `data/`, `notebooks/`, and `outputs/` from the start. The starter notebook is `notebooks/start.ipynb`; environment details are saved as `outputs/setup.json`.
- **Reason:** The user wants clean modular code and room to expand the project, while avoiding empty folders and extra setup files that do not yet serve a purpose.
- **Consequence:** The initial and expanded layouts are recorded in `architecture.md`.

## 2026-10-29 — Record the available remote compute environment

- **Choice:** Use the user's remote Jupyter environment for training, and use its CPU for the planned CPU inference comparison.
- **Observed environment from user-provided command output:** Python 3.10.12; PyTorch 2.4.0a0+f70bd71a48.nv24.06; TorchVision 0.19.0a0; CUDA available through PyTorch; NVIDIA H100 80GB HBM3 with a visible MIG allocation of about 40 GB; CUDA runtime 12.5; driver 535.161.08; Intel Xeon Platinum 8480C.
- **Container limits shown:** 224 logical CPUs are visible, but the cgroup CPU quota is 1,500,000 microseconds per 100,000-microsecond period (15 CPU-equivalents); the cgroup memory limit is 128,849,018,880 bytes (120 GiB). Host-level `free` output should not be reported as the container limit.
- **Consequence:** Save a fresh environment record from the notebook before experiments. For CPU benchmarks, explicitly set and record thread count and note the container quota. Confirm the quantization backend on this PyTorch build before relying on dynamic INT8 results.

## 2026-10-29 — Workspace handoff documentation created

- **Choice:** Treat pruning vs. quantization on Fashion-MNIST as the active project and document it at the workspace root.
- **Reason:** The user selected the previously discussed third project and requested detailed continuation documentation.
- **Workspace observation:** This folder currently contains `KFall_dataset/` and `SisFall_dataset/`, but no compression-study code or Fashion-MNIST cache was found during documentation setup.
- **Consequence:** Existing fall-detection data is explicitly marked unrelated and must be preserved. All planned architecture/results remain proposals until implementation and runs happen.
