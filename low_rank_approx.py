import torch

def low_rank_approx(n_feature,\
                    Q_mask,\
                    M_d_0,\
                    low_rank_k):          
    tril_idx=torch.tril_indices(n_feature,n_feature)
    Cholesky_U_0=torch.zeros(n_feature,n_feature)
    Cholesky_U_0[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]=M_d_0
    M0=Cholesky_U_0@Cholesky_U_0.T
    factor_for_diag=torch.trace(M0)/n_feature
    M=M0/factor_for_diag
    M_value,M_vector=torch.linalg.eig(M)
    M_value_sort,M_value_indices=torch.real(M_value).sort(descending=True)
    M_lr_value=M_value_sort[0:low_rank_k]
    M_lr_vector=torch.real(M_vector)[:,M_value_indices[0:low_rank_k]]
    M_rec=M_lr_vector@torch.diag(torch.sqrt(M_lr_value))
    return M_rec