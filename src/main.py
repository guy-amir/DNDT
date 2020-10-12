import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from functools import reduce
from model import DNDT

# iris = datasets.load_iris()
# X = torch.tensor(iris.data)[:,2:4]
# y = torch.tensor(iris.target)

import sklearn

# n_samples = 200
# noisy_moons = sklearn.datasets.make_moons(n_samples=n_samples, noise=.15)
# X,y = noisy_moons

# X = torch.tensor(X)
# y = torch.tensor(y)

# n_features = X.shape[1]
# n_classes = y.max()+1



# epochs = 1000
# lr = 0.01
# n_bins=2
# temperature = 1

def dndt_trainer(X,y,epochs=1000,lr=0.01,n_features=2,n_classes=2,n_bins=2,temperature=1):
    dndt = DNDT(n_features,n_bins,n_classes,temperature)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([dndt.beta]+[dndt.leaves2classes], lr=0.01)
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = dndt.forward(X)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f"loss: {loss.detach().numpy()}")
            correct = (y_pred.argmax(1) == y).float().sum()
            print(f"accuracy: {correct/len(y)}")
    return dndt


# dndt_runner(X,y,epochs,lr,n_bins,temperature)