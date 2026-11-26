# Instructions for continuing this project

## Project

A reproducible study comparing unstructured magnitude pruning and dynamic INT8 quantization on Fashion-MNIST. The contribution is a careful comparison of accuracy, actual saved model size, measured sparsity and measured CPU latency. Do not describe the project as a new compression method.

Read these before making changes:

1. `docs/overview.md` for scope and status.
2. `docs/architecture.md` for the code layout and output files.
3. `docs/protocol.md` before changing data splits, methods or metrics.
4. `docs/results.md` for the current results.
5. `docs/tasks.md` for open work.
6. `docs/decisions.md` and `docs/related-work.md` before changing scope or making novelty claims.

## Workspace

- Everything runs from the project folder. Paths come from `src/config.py` (`root`, `datadir`, `outdir`); never hard-code machine-specific paths.
- Environment: `.venv` created from `requirements.txt`. Inspect the installed versions before recommending upgrades.
- `notebooks/start.ipynb` is the only main notebook. It imports `src/` and holds no reusable logic.
- Each `src/` module has one responsibility (see `architecture.md`). Add a small new module when a real new responsibility does not fit an existing one.
- `data/` is the dataset cache. `outputs/` is generated and can be recreated by running the notebook.
- Keep the README short. Technical detail belongs in the relevant document.

## Coding and explanations

- Prefer short, readable code. Avoid dense one-liners and unnecessary abstractions.
- Use lowercase names without underscores where practical (`trainset`, `prunemodel`, `datadir`). Keep library API names as required (`FashionMNIST`, `state_dict`).
- When giving code to the user, give small snippets in run order. Name the exact destination (module, or notebook section) and explain the important lines and choices, including terms such as seeds.
- When code changes, update `docs/code-notes.md` so its explanation matches the code.

## Experiment integrity

- The official test set is for final evaluation only, never for training, checkpoint choice or tuning.
- Derive every compressed variant independently from the same seed's baseline.
- Report measured sparsity, saved artifact bytes and CPU latency as separate results.
- Do not claim pruning made a model smaller or faster unless the saved file or benchmark shows it.
- Every reported number must come from a saved file in `outputs/`. After re-running, update `results.md`, the README result lines and the notebook findings from `outputs/metrics/summary.csv`.
- Record meaningful progress in `docs/progress.md`. Record any protocol change in `docs/decisions.md` before treating its results as final.
- Use real dates in logs. Never backdate entries.
- Structured pruning, fine-tuning, extra datasets and new quantization methods are optional extensions. Discuss them with the user before adding.

## Recorded environment

The final run used Apple M2 (8 logical CPUs), macOS 26.3.1, Python 3.11.15, PyTorch 2.15.0, TorchVision 0.29.2 and the `qnnpack` INT8 engine (`outputs/setup.json`). The October pilot used a remote server: Python 3.10.12, PyTorch 2.4.0a0, NVIDIA H100 MIG, Intel Xeon Platinum 8480C with a 15-CPU container quota, and the `x86` engine. Latency results are machine-specific; re-record `setup.json` whenever the machine changes.
