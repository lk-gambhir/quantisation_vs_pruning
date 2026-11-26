"""Common test-set evaluation used for every variant."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def predict(model, loader):
    """Return predicted and true labels for every example in loader order (CPU)."""
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            predictions.append(model(images).argmax(1))
            labels.append(targets)
    return torch.cat(predictions).numpy(), torch.cat(labels).numpy()


def scores(predictions, labels):
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macrof1": float(f1_score(labels, predictions, average="macro")),
    }


def perclass(predictions, labels, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(classes))), zero_division=0)
    return [{"class": classes[i], "precision": precision[i], "recall": recall[i], "f1": f1[i],
             "support": int(support[i])} for i in range(len(classes))]


def confusion(predictions, labels, classes):
    return confusion_matrix(labels, predictions, labels=list(range(len(classes))))


def comparepredictions(basepred, variantpred, labels):
    """Count test examples that change between the FP32 baseline and a variant."""
    basecorrect = basepred == labels
    variantcorrect = variantpred == labels
    return {
        "newerrors": int(np.sum(basecorrect & ~variantcorrect)),
        "fixed": int(np.sum(~basecorrect & variantcorrect)),
        "bothwrong": int(np.sum(~basecorrect & ~variantcorrect)),
        "disagree": int(np.sum(basepred != variantpred)),
    }
