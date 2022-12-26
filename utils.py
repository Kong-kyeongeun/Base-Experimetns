import torch
import numpy as np


def PearsonCorrelation(g_w, ce, recip=False):
    assert len(g_w.size())==2 and len(ce.size())==1, 'Necessitate checking out dimensions'

    g_w = g_w.mean(0).squeeze(0)
    if recip:
        g_w = torch.reciprocal(g_w)
    # True recip, desire to p -> +1
    # False recip, desire to p -> -1

    # We can get pearson coefficient per a batch
    g_w = (g_w - torch.mean(g_w)) + 1e-6
    ce = (ce - torch.mean(ce)) + 1e-6

    pc = torch.sum(g_w*ce) / (torch.sqrt(torch.sum(g_w**2)) * torch.sqrt(torch.sum(ce**2)))
    return pc

def SpearmanCorrelation(g_w, ce, recip=False):
    assert len(g_w.size())==2 and len(ce.size())==1, 'Necessitate checking out dimensions'

    g_w = g_w.mean(0).squeeze(0)
    if recip:
        g_w = torch.reciprocal(g_w)

    g_w = torch.FloatTensor(pd.DataFrame(g_w).rank().to_numpy()).t()
    ce = torch.FloatTensor(pd.DataFrame(ce).rank().to_numpy()).t()

    g_w = (g_w - torch.mean(g_w)) + 1e-6
    ce = (ce - torch.mean(ce)) + 1e-6

    rho = torch.sum(g_w*ce) / (torch.sqrt(torch.sum(g_w**2)) * torch.sqrt(torch.sum(ce**2)))

    return rho