"""Fashion-MNIST loading, the seeded train/validation split, and data loaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import config


def loaddatasets():
    """Load the official train and test partitions with pixels scaled to [0, 1]."""
    convert = transforms.ToTensor()
    trainset = datasets.FashionMNIST(config.datadir, train=True, download=True, transform=convert)
    testset = datasets.FashionMNIST(config.datadir, train=False, download=True, transform=convert)
    return trainset, testset


def splitindices(count, seed):
    """Shuffle 0..count-1 with a seeded generator; the first 90% train, the rest validate.

    This gives the same split as torch.utils.data.random_split with the same generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(count, generator=generator).tolist()
    traincount = count - int(count * config.valfraction)
    return order[:traincount], order[traincount:]


def makeloaders(trainset, testset, seed):
    """Build the three loaders for one seed. Only the training loader is shuffled."""
    trainindices, valindices = splitindices(len(trainset), seed)
    ordergen = torch.Generator()
    ordergen.manual_seed(seed)
    trainloader = DataLoader(Subset(trainset, trainindices), batch_size=config.batchsize,
                             shuffle=True, generator=ordergen)
    valloader = DataLoader(Subset(trainset, valindices), batch_size=config.batchsize)
    testloader = DataLoader(testset, batch_size=config.batchsize)
    split = {"seed": seed, "train": trainindices, "validation": valindices}
    return trainloader, valloader, testloader, split
