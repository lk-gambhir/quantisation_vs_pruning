"""Baseline training with checkpoint selection by validation loss."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.model import MLP


def setseed(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def runepoch(model, loader, device, optimizer=None):
    """One pass over a loader. Trains when an optimizer is given, otherwise only measures."""
    training = optimizer is not None
    model.train(training)
    lossfn = nn.CrossEntropyLoss(reduction="sum")
    totalloss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = lossfn(logits, labels)
            if training:
                optimizer.zero_grad()
                (loss / labels.size(0)).backward()
                optimizer.step()
            totalloss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return totalloss / total, correct / total


def trainbaseline(seed, trainloader, valloader, device):
    """Train a fresh MLP and return the epoch with the lowest validation loss.

    The test set is never used here.
    """
    setseed(seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningrate)
    history = []
    beststate, bestloss, bestepoch = None, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        trainloss, trainacc = runepoch(model, trainloader, device, optimizer)
        valloss, valacc = runepoch(model, valloader, device)
        history.append({"seed": seed, "epoch": epoch, "trainloss": trainloss, "trainacc": trainacc,
                        "valloss": valloss, "valacc": valacc})
        if valloss < bestloss:
            beststate = copy.deepcopy(model.state_dict())
            bestloss, bestepoch = valloss, epoch
    model.load_state_dict(beststate)
    return model.cpu().eval(), history, bestepoch
