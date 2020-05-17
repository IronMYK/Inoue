import torch

def loss_w1(a, b, gamma):
    ret = torch.abs((a-b) * (1 + gamma * (1 - b)))
    return torch.mean(ret)