import numpy as np
import torch as t
import matplotlib.pyplot as plt
from sklearn import datasets
from functools import reduce
from model import DNDT

iris = datasets.load_iris()
X = t.tensor(iris.data)[:,2:4]
y = t.tensor(iris.target)

n_features = X.shape[1]
n_classes = y.max()+1




epochs = 10000
dndt = DNDT(n_features,1,n_classes,temperature=3)
loss_function = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam([dndt.beta]+[dndt.leaves2classes], lr=0.1)
for i in range(epochs):
    optimizer.zero_grad()
    y_pred = dndt.forward(X)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    if i % 200 == 0:
        print(loss.detach().numpy())