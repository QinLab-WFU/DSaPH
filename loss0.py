from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

# from BaseLine.miner import TripletMarginMiner
# from BaseLine.utils import distance


class DFPHLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.theta = config['theta']
        self.eta = config['eta']
        self.gamma = config['gamma']
        self.n_bits = config['nbit']

    def forward(self, img_hash, lab_hash, labels):
        logit = img_hash.mm(lab_hash.t())
        # logit = lab_hash
        nits = labels.sum(1)
        batch_size = labels.shape[0]

        # Eq. 6
        # code        -> paper
        # gamma=0.3   -> eta=0.1
        # sigma=0.3*k -> theta=0.3
        # lamda=0.1   -> gamma=0.1
        numerator = torch.exp(((logit * labels).sum(1) / nits - self.theta * self.n_bits) * self.eta)
        denominator = torch.exp(logit * (1 - labels) * self.eta).sum(1) + numerator
        sem_sim_loss = -(torch.log(numerator / denominator)).sum() / batch_size

        # Eq. 7
        quan_loss = (img_hash.detach().sign() - img_hash).pow(2).sum() / batch_size

        return sem_sim_loss + quan_loss * self.gamma
