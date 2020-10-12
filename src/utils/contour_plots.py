print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import torch



def plot_results(X,y,model,image_name=None):
    fig, sub = plt.subplots(1,1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    ax = sub

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_2d_function(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_xticks(())
    ax.set_yticks(())
    # ax.set_title(title)
    if image_name is not None:
        plt.savefig(image_name)

def plot_2d_function(ax, model, xx,yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    txx = torch.tensor(xx.ravel()).unsqueeze(1).to(device=device)
    tyy = torch.tensor(yy.ravel()).unsqueeze(1).to(device=device)
    samps = torch.cat((txx,tyy),1).float()
    
    Z = model(samps.double()).detach().to(device='cpu').numpy()
    Z = Z[:,0].reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    margin = 0.1
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    return xx, yy
