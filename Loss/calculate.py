import torch
import numpy as np
from Loss.ILoss import SupConLoss
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.sparse as sp


def calculate_loss(x, y, temperature, Lambda, alpha, W=1):
    loss = alpha * loss_dim_label_mask(x, y, Lambda) + loss_instance(x, y, temperature, W)
    return loss


def loss_dim_label_mask(x, y, Lambda):
    x = x - x.mean(0)
    c = (x / torch.where(torch.norm(x, dim=0) != 0, torch.norm(x, dim=0), 1).expand_as(x)).T @ (
            x / torch.where(torch.norm(x, dim=0) != 0, torch.norm(x, dim=0), 1).expand_as(x))
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = Lambda * off_diag
    return loss


def loss_instance(x, y, temperature, W):
    criterion = SupConLoss(temperature=temperature)
    T = abs(W)
    x = x.unsqueeze(1)
    loss = criterion(features=x, labels=y, adj=[])
    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def class_sim(x, y):
    inner_sim = 0
    intra_sim = 0
    x_lists = {}
    # divide by class
    for i in range(x.shape[0]):
        if x_lists.get(y[i].item()) is None:
            x_lists[y[i].item()] = x[i].unsqueeze(0)
        else:
            x_lists[y[i].item()] = torch.cat([x_lists[y[i].item()], x[i].unsqueeze(0)], dim=0)

    # calculate cosine similarity inner class
    lists = list(x_lists.items())

    # calculate pair
    for i in range(len(lists)):
        x1 = lists[i][1]
        for j in range(i + 1, len(lists)):
            x2 = lists[j][1]
            from sklearn.metrics.pairwise import cosine_similarity
            inner_sim_t = cosine_similarity(np.array(x1.detach().cpu().numpy()), np.array(x2.detach().cpu().numpy()))
            inner_sim += inner_sim_t.sum() / (x1.shape[0] * x2.shape[0])

    inner_sim = inner_sim / (len(lists) * (len(lists) - 1) / 2)

    # calculate cosine similarity intra class
    for key, value in x_lists.items():
        value = value.cpu().detach().numpy()

        intra_sim_t = cosine_similarity(np.array(value), np.array(value))
        np.fill_diagonal(intra_sim_t, 0)
        intra_sim += (intra_sim_t.sum()) / (value.shape[0] * (value.shape[0] - 1))

    intra_sim = intra_sim / len(x_lists.items())

    r_class = intra_sim - inner_sim

    return inner_sim, intra_sim, r_class

