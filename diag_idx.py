import torch 

def diag_idx(n_feature):
    diag_indices0=torch.tensor(range(0,n_feature)).reshape((1,n_feature))
    diag_indices=torch.cat((diag_indices0,diag_indices0),dim=0)
    return diag_indices