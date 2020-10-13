import torch

class parameters():
    def __init__(self):

        #General parameters
        self.output_path = './results/DL_layers/analysis'
        self.archive_path = './archive'
        self.save = True

        #Computational parameters:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.n_samples = 200

        304526254