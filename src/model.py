import torch
from functools import reduce

class DNDT(torch.nn.Module):
    def __init__(self, n_features, n_cuts, n_classes,temperature):
        super(DNDT, self).__init__()
        self.n_bins = n_cuts+1
        self.n_cuts = n_cuts
        self.beta = torch.rand(self.n_cuts)
        self.leaves2classes = torch.rand((self.n_bins**n_features,n_classes))
        self.leaves2classes.requires_grad = True
        self.beta.requires_grad = True
        self.temperature = temperature
        
    def pi_maker(feature,beta,temperature=0.1):
        n_bins = len(beta)
        w = torch.reshape(torch.linspace(1,n_bins,n_bins),[-1,1]) #make constant or something later
        torch.Variable(w, requires_grad=False)
        beta, _ = torch.sort(beta)
        beta[0] = 0
        b = torch.cumsum(-beta,0)
        pi = torch.reshape(torch.softmax((w*feature+b)/temperature,0),[-1,1])
        return pi
    
    def beta2b(self):
        beta, _ = torch.sort(self.beta)
        b = torch.cumsum(-beta,0)
        return torch.cat([torch.zeros(1), b], 0).double()
         
    def forward(self, x):

        x = x.unsqueeze(2)
        w = torch.reshape(torch.linspace(1,self.n_bins,self.n_bins),[1,-1]).double()
        b = self.beta2b()
        xw = torch.matmul(x,w)
        
        sigma = torch.sigmoid((xw+b)/self.temperature)
        leaves = torch.zeros((sigma.size(0),sigma.size(2)**sigma.size(1)))
        for i,sample in enumerate(sigma):
            feature_list = [feature.unsqueeze(1) for feature in sample]
            leaves[i] = reduce(kronecker,feature_list).squeeze(1)
        
        y_pred = leaves@self.leaves2classes
        
        return y_pred

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))