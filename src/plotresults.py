"""Figures. Every function reads saved result tables; none of them re-evaluates a model."""
import matplotlib.pyplot as plt
import numpy as np

colors = {"fp32": "#1f77b4", "pruned30": "#ffbb78", "pruned50": "#ff7f0e", "pruned70": "#d62728",
          "pruned90": "#7f0000", "int8": "#2ca02c", "pruned50int8": "#9467bd"}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def learningcurves(history, bestepochs, path):
    fig, (lossax, accax) = plt.subplots(1, 2, figsize=(12, 4))
    for seed, rows in history.groupby("seed"):
        line, = lossax.plot(rows.epoch, rows.trainloss, "--", alpha=0.7)
        lossax.plot(rows.epoch, rows.valloss, "-", color=line.get_color(), label=f"seed {seed}")
        lossax.axvline(bestepochs[seed], color=line.get_color(), linestyle=":", alpha=0.6)
        accax.plot(rows.epoch, rows.trainacc, "--", color=line.get_color(), alpha=0.7)
        accax.plot(rows.epoch, rows.valacc, "-", color=line.get_color(), label=f"seed {seed}")
    lossax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
               title="Loss (dashed = train, solid = validation, dotted = chosen epoch)")
    accax.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (dashed = train, solid = validation)")
    for ax in (lossax, accax):
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, path)
    return fig


def qualitysize(results, summary, tolerance, path):
    """Accuracy against saved artifact size, one small dot per seed and a large marker per mean."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant, rows in results.groupby("variant", sort=False):
        ax.scatter(rows.bytes / 1000, rows.accuracy, color=colors[variant], alpha=0.4, s=25)
        mean = summary.loc[variant]
        ax.errorbar(mean.bytes / 1000, mean.accuracy, yerr=mean.accuracysd, fmt="o", ms=10,
                    color=colors[variant], capsize=4, label=variant)
    baseline = summary.loc["fp32", "accuracy"]
    ax.axhspan(baseline - tolerance / 100, baseline + 0.01, color="green", alpha=0.08,
               label=f"within {tolerance:g} pp of FP32")
    ax.set(xlabel="Saved artifact size (kB, 1 kB = 1000 bytes)", ylabel="Test accuracy",
           title="Accuracy vs actual artifact size (mean ± SD over seeds; dots = seeds)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="center")
    save(fig, path)
    return fig


def qualitylatency(latency, summary, path):
    """Accuracy against median CPU latency for each batch size."""
    batches = sorted(latency.batchsize.unique())
    fig, axes = plt.subplots(1, len(batches), figsize=(13, 5))
    for ax, batchsize in zip(axes, batches):
        rows = latency[latency.batchsize == batchsize].groupby("variant", sort=False)
        for variant, group in rows:
            ax.errorbar(group.p50ms.mean(), summary.loc[variant, "accuracy"],
                        xerr=group.p50ms.std(ddof=1), fmt="o", ms=9, capsize=4,
                        color=colors[variant], label=variant)
        ax.set(xlabel="Median latency per batch (ms)", ylabel="Test accuracy",
               title=f"Batch size {batchsize} (error bar = SD of p50 across seeds)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    save(fig, path)
    return fig


def pruningcurve(results, levels, path):
    """Measured sparsity, accuracy and macro F1 against the pruning target."""
    names = ["fp32"] + [f"pruned{round(level * 100)}" for level in levels]
    targets = [0] + [round(level * 100) for level in levels]
    grouped = results[results.variant.isin(names)].groupby("variant")
    mean, sd = grouped.mean(numeric_only=True).loc[names], grouped.std(numeric_only=True).loc[names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(targets, mean.accuracy, yerr=sd.accuracy, marker="o", capsize=4, label="Accuracy")
    ax.errorbar(targets, mean.macrof1, yerr=sd.macrof1, marker="s", capsize=4, label="Macro F1")
    ax.plot(targets, mean.sparsity, "k--", marker="^", label="Measured weight sparsity")
    ax.set(xlabel="Pruning target (% of Linear weights)", ylabel="Value (0 to 1)",
           title="Quality and measured sparsity vs pruning target (mean ± SD over seeds)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)
    return fig


def confusionplot(matrix, classes, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(image, ax=ax)
    save(fig, path)
    return fig


def perclassf1(perclass, variants, path):
    """Mean per-class F1 across seeds for selected variants."""
    table = perclass[perclass.variant.isin(variants)].groupby(["class", "variant"]).f1.mean().unstack()
    table = table.loc[perclass["class"].unique(), variants]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.8 / len(variants)
    for i, variant in enumerate(variants):
        ax.bar(np.arange(len(table)) + i * width, table[variant], width, label=variant,
               color=colors[variant])
    ax.set_xticks(np.arange(len(table)) + 0.4 - width / 2, table.index, rotation=30, ha="right")
    ax.set(ylabel="F1 (mean over seeds)", title="Per-class F1 by variant")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=len(variants), loc="lower center")
    save(fig, path)
    return fig


def weighthist(layers, thresholds, path):
    """|weight| distribution of each baseline layer with the global pruning cut-offs marked."""
    fig, axes = plt.subplots(1, len(layers), figsize=(15, 4))
    for ax, (name, weight) in zip(axes, layers):
        ax.hist(weight.abs().flatten().numpy(), bins=100, color="gray")
        for label, value in thresholds.items():
            ax.axvline(value, color=colors[label], linestyle="--", label=f"{label} cut-off")
        ax.set(title=f"{name}  {tuple(weight.shape)}", xlabel="|weight|", ylabel="Count")
        ax.set_yscale("log")
    axes[-1].legend()
    fig.suptitle("Baseline |weight| per layer (seed 42) and global L1 pruning thresholds")
    fig.tight_layout()
    save(fig, path)
    return fig


def layersparsityplot(layerrows, path):
    table = layerrows.groupby(["variant", "layer"]).sparsity.mean().unstack()
    order = [v for v in colors if v in table.index and v != "fp32"]
    table = table.loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(table.columns)
    for i, layer in enumerate(table.columns):
        ax.bar(np.arange(len(table)) + i * width, table[layer], width, label=layer)
    ax.set_xticks(np.arange(len(table)) + 0.4 - width / 2, table.index)
    ax.set(ylabel="Fraction of zero weights", title="Measured sparsity per layer (mean over seeds)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, path)
    return fig


def predictionhist(counts, classes, path):
    """How often each class is predicted, per variant (one seed)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / len(counts)
    for i, (variant, values) in enumerate(counts.items()):
        ax.bar(np.arange(len(classes)) + i * width, values, width, label=variant, color=colors[variant])
    ax.axhline(1000, color="black", linestyle=":", label="true count per class")
    ax.set_xticks(np.arange(len(classes)) + 0.4 - width / 2, classes, rotation=30, ha="right")
    ax.set(ylabel="Number of test predictions", title="Predicted-class counts (seed 42)")
    ax.legend()
    save(fig, path)
    return fig


def examplegrid(testset, rows, classes, title, path):
    """Show test examples by ID with their true, FP32 and variant labels."""
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.5))
    for ax, row in zip(axes.flat, rows.itertuples()):
        image, _ = testset[row.index]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"#{row.index} true {classes[row.label]}\nfp32 {classes[row.fp32]}\n"
                     f"new {classes[row.variantpred]}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path)
    return fig
