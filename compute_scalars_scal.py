import torch

def compute_scalars_scal(initial_H,n_sample,evector):
    evector_inv=torch.div(1,evector)
    scaled_M=torch.multiply(torch.multiply(evector_inv,initial_H),\
                     evector.T)
    scaled_factors=torch.multiply(torch.multiply(evector_inv,\
                     torch.ones(n_sample+2,n_sample+2)),\
                     evector.T)
    return scaled_M,scaled_factors