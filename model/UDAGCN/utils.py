import torch

def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot
