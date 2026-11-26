# Task list

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`. A task is `DONE` only when its acceptance criteria are met and the evidence is recorded in `progress.md`. History before 2026-11-24 (the remote pilot snippets and handoffs) is in `progress.md`.

## Summary

| ID | Task | Status | Evidence |
|---|---|---|---|
| T00 | Confirm teacher brief and deliverables | TODO | — |
| T01 | Environment and project skeleton | DONE | `requirements.txt`, `outputs/setup.json`, `architecture.md` |
| T10 | Load and document Fashion-MNIST | DONE | `outputs/dataset-summary.json`, `figures/samples.png` |
| T11 | Deterministic split and loaders | DONE | `src/data.py`, `outputs/runs/seed*/split.json` |
| T20 | Model and configuration | DONE | `src/model.py`, `src/config.py`, `outputs/runs/config.json` |
| T21 | Baseline training with validation-loss checkpoint | DONE | `src/train.py`, `outputs/runs/seed*/history.csv`, `figures/learning-curves.png` |
| T22 | Baseline test result | DONE | `metrics/results.csv`, `metrics/confusion/seed*-fp32.csv` |
| T23 | Verify the 90% pruning result (teacher request) | DONE | Notebook Sections 11 and 13, `results.md` §3 |
| T30 | Artifact save/reload utilities | DONE | `src/compress.py` (`saveartifact`, `loadartifact`), reload assert in notebook §6 |
| T31 | Magnitude pruning 30/50/70/90% | DONE | `metrics/results.csv`, `metrics/layersparsity.csv` |
| T32 | INT8 backend check and dynamic INT8 | DONE | `setup.json` (`qnnpack`), `checkpoints/seed*/int8/` |
| T33 | Combined 50% prune → INT8 | DONE | `checkpoints/seed*/pruned50int8/` |
| T40 | Common test evaluation | DONE | `src/evaluate.py`, `metrics/perclass.csv`, `metrics/predictions/` |
| T41 | Size and sparsity measurement | DONE | `bytes`, `bitsperparam`, `ratio` and `sparsity` columns in `metrics/summary.csv` |
| T42 | CPU latency benchmark | DONE | `src/benchmark.py`, `metrics/latency.csv`, `metrics/latency-raw.csv` |
| T43 | Repeated seeds | DONE (3 of 5) | Seeds 42, 123, 2026 in every metrics file |
| T44 | Seeds 2027 and 2028 (five-seed target) | TODO, optional | — |
| T50 | Final plots | DONE | `outputs/figures/` (12 figures) |
| T51 | Error and trade-off analysis | DONE | `results.md` §3, `metrics/failures-seed42.csv`, `metrics/comparisons.csv` |
| T52 | Final report in the teacher's format | TODO | Draft content in `results.md` |
| T53 | Presentation / demo | TODO | — |

## Open tasks

### T00 — Confirm the teacher's brief

- **Actions:** Record the exact assignment wording, required sections, required tools, number of experiments or figures, format and deadline in `docs/teacher-requirements.md`. Confirm that comparing pruning and quantization in one project is acceptable.
- **Acceptance:** Every requirement is quoted from the brief or labelled as a student decision. Nothing is invented.

### T44 — Reach five seeds (optional)

- **Actions:** Set `seeds = [42, 123, 2026, 2027, 2028]` in `src/config.py`, re-run `notebooks/start.ipynb`, then update `results.md`, the README result lines and the notebook findings from the new `summary.csv`.
- **Acceptance:** All seven variants are complete for every seed, and the stated seed count matches the `seeds` column in `summary.csv`.
- **Cost:** About 2 extra minutes of compute.

### T52 — Final report

- **Actions:** Turn `results.md` into the teacher's required format, covering problem, related work (`related-work.md`), data, method (`protocol.md`), results, analysis, limitations, conclusion and references (`references.md`). Use only figures from `outputs/figures/` and numbers from `outputs/metrics/`.
- **Acceptance:** Every number matches a saved file. Claims stay limited to this model, dataset and machine. The pilot runs are labelled as exploratory.

### T53 — Presentation / demo

- **Actions:** Slides covering the question, the pipeline diagram (`architecture.md`), the three trade-off figures, the 90% collapse and the limitations. Prepare a live demo: run one reloaded artifact on a test image and show where its metrics row comes from.
- **Be ready to answer:**
  - Why don't zero weights shrink the file or speed up inference?
  - Why is INT8 slower on this laptop?
  - How was test-set leakage avoided?
  - Why choose the checkpoint by validation loss?
  - Why does global pruning cut layers unequally?

## Possible extensions (only with the teacher's agreement)

- Fine-tuning after pruning at 70% and 90%, using one fixed recipe for every level and reported as a separate experiment. In the 2026-11-24 verification, a 2-epoch side check recovered 90% pruning to about 0.84 accuracy.
- Structured (whole-neuron) pruning, which physically shrinks the matrices and can reduce dense latency.
- CSR sparse storage size, including the index overhead.
- Latency on an x86 CPU with the `fbgemm`/`x86` INT8 engine.
