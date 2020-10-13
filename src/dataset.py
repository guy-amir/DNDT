import torch
from torch.utils.data import DataLoader, Dataset, random_split
import sklearn.datasets

def moons_dl(prms):
    
    init_dataset = moonsDataset(prms)
    params = {'batch_size': prms.n_samples,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(prms.n_samples*0.8), prms.n_samples-int(prms.n_samples*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    dataloader = {'train':train_loader,'val':test_loader}

    return train_dataset,valid_dataset,dataloader

class moonsDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        
        noisy_moons = sklearn.datasets.make_moons(n_samples=prms.n_samples, noise=.15)
        X,Y = noisy_moons

        self.samples = torch.tensor(X)
        self.labels = torch.tensor(Y)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y