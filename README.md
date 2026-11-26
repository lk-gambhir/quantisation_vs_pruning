# Pruning vs INT8 Quantization on Fashion-MNIST

A controlled study of two ways to compress a small image classifier: unstructured magnitude pruning and dynamic INT8 quantization. It measures accuracy, saved file size and CPU latency as separate outcomes, over three paired training seeds. This is a careful comparison, not a new compression method.

## Main result

Over 3 seeds on an Apple M2 CPU:

- **INT8** makes the saved model **3.91× smaller** (943,357 → 240,987 bytes) with no accuracy change (+0.03 ± 0.07 pp). On this CPU it runs about **3.5–5× slower**.
- **Pruning** creates exactly the requested share of zero weights. The file size and latency stay the same as FP32. Accuracy holds up to 50% pruning (−0.72 pp), then drops to −6.4 pp at 70% and −47.8 pp at 90%.

The full tables and figures are in [`docs/results.md`](docs/results.md).

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/start.ipynb   # or open it and Run All
```

Fashion-MNIST downloads to `data/` on the first run. A full run takes about 1–4 minutes on a laptop CPU and rewrites everything under `outputs/`.

## Layout

```text
notebooks/start.ipynb   main notebook: runs the whole study and shows results
src/                    reusable code (config, data, model, train, compress, evaluate, benchmark, plotresults)
outputs/                generated: setup record, checkpoints, metrics CSVs, figures
docs/                   overview, protocol, architecture, results, decisions, progress, code notes
```

## Documents

| Document | Contents |
|---|---|
| [`docs/overview.md`](docs/overview.md) | Research question, scope, deliverables |
| [`docs/protocol.md`](docs/protocol.md) | Frozen experiment rules: split, seeds, methods, metrics, decision rule |
| [`docs/architecture.md`](docs/architecture.md) | Code layout, data flow, output files |
| [`docs/results.md`](docs/results.md) | Final results, figures, analysis, limitations |
| [`docs/code-notes.md`](docs/code-notes.md) | Explanation of each module and notebook section |
| [`docs/decisions.md`](docs/decisions.md) | Dated log of protocol and scope decisions |
| [`docs/progress.md`](docs/progress.md) | Dated, evidence-based work log |
| [`docs/tasks.md`](docs/tasks.md) | Task list and status |
| [`docs/related-work.md`](docs/related-work.md), [`docs/references.md`](docs/references.md) | Prior work and citations |
