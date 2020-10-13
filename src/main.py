# from params import parameters
# from dataset import moons_dl

# prms = parameters()

# trainset, testset, dataloader = moons_dl(prms)

# X_train = trainset[:][0].numpy()
# y_train = trainset[:][1].numpy()
# X = testset[:][0].numpy()
# y = testset[:][1].numpy()

from train import dndt_trainer
import sklearn.datasets
import torch
import matplotlib.pyplot as plt
import numpy as np
import utils.contour_plots
from itertools import cycle, islice

from sklearn.model_selection import train_test_split

from params import parameters
from dataset import moons_dl

prms = parameters()

trainset, testset, dataloader = moons_dl(prms)

dataset_noise = .15
n_samples = 100
epochs = 10
n_bins = 4

model_log,dndt = dndt_trainer(dataloader,n_bins=n_bins,epochs=epochs)