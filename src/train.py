import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from functools import reduce
from model import DNDT
import time
import copy

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

def dndt_trainer(dataloader,epochs=1000,lr=0.01,n_features=2,n_classes=2,n_bins=2,temperature=1, save_every=200):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DNDT(n_features,n_bins,n_classes,temperature).to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # X_train.to(device)
    # X_test.to(device)
    # y_train.to(device)
    # y_test.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([model.beta]+[model.leaves2classes], lr=lr)

    model_loss = {'train':[],'val':[]}
    model_acc = {'train':[],'val':[]}
    # train_loss = []
    # train_acc = []
    # val_loss = []
    # val_acc = []

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                dataset_sizes = len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes

            model_loss[phase].append(epoch_loss)
            model_acc[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # optimizer.zero_grad()
        # y_pred = dndt.forward(X_train).to(device)
        # loss = loss_function(y_pred, y_train)
        # loss.backward()
        # optimizer.step()
        # if i % save_every == 0:

        #     train_loss.append(loss.cpu().detach().numpy())
        #     print(f"train loss: {train_loss[epoch]}")

        #     correct = (y_pred.argmax(1) == y).float().sum()
        #     train_acc.append(correct/len(y))
        #     print(f"train acc: {train_acc[epoch]}")

        #     val_loss.append(loss.cpu().detach().numpy())
        #     print(f"val loss: {val_loss[epoch]}")

        #     correct = (y_pred.argmax(1) == y).float().sum()
        #     val_acc.append(correct/len(y))
        #     print(f"val acc: {val_acc[epoch]}")

        time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    model_log = {'train_loss':model_loss['train'],'train_acc':model_acc['train'],'val_loss':model_loss['val'],'val_acc':model_acc['val']}
    return model_log,model


# dndt_runner(X,y,epochs,lr,n_bins,temperature)

# def train_model(model, criterion, optimizer, scheduler, epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(epochs):
#         print('Epoch {}/{}'.format(epoch, epochs - 1))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloader[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model
