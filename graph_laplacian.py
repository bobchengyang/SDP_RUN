import torch

def graph_laplacian(n_sample,c,M):
    W=torch.exp(-torch.sum(torch.multiply(torch.matmul(c,M),c),axis=1))
    W_reshape=W.reshape((n_sample,n_sample))
    W_u=W_reshape-torch.eye(n_sample)
    D=torch.diag(W_u.sum(axis=1))
    L=D-W_u
    return L