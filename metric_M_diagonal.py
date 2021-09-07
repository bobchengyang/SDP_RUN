import torch
from graph_construction import graph_construction

def metric_M_diagonal(M_normalizer,feature,\
                      b_ind,\
                      label,\
                      n_feature,\
                      M_d_in,\
                      n_train,\
                      metric_M_step,Q_mask,optimizer_M,\
                      M_rec,\
                      low_rank_yes_no):
    
    if low_rank_yes_no==0:
        tril_idx=torch.tril_indices(n_feature,n_feature)
        Cholesky_U_0=torch.zeros(n_feature,n_feature)
        Cholesky_U_0[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]=M_d_in
        M0=Cholesky_U_0@Cholesky_U_0.T
    else:
        M0=M_rec@M_rec.T
        
    factor_for_diag=torch.trace(M0)/M_normalizer
    M=M0/factor_for_diag
    

        
    # v = Variable(M_d_in.reshape(n_feature), requires_grad=True)
    # M_0=torch.diag(v)
    feature_train=feature[b_ind,:]
    L=graph_construction(feature_train, n_train, n_feature, M)
    metric_M_obj=torch.matmul(torch.matmul(label[b_ind].reshape(1,n_train),L),\
                                label[b_ind].reshape(n_train,1))
    
    metric_M_obj.backward()
    optimizer_M.step() 
    # print(metric_M_obj)
    # projection
    # M_d=F.relu(M_d_in-metric_M_step*v.grad)
    # trace(M) <= n_feature
    # while M_d.sum()>n_feature:
    #     try_num=(M_d.sum()-n_feature)/M_d.count_nonzero()
    #     M_d=F.relu(M_d-try_num)
    # M_d_out=M_d.reshape(n_feature)
    # M_d_out=torch.multiply(M_d,n_feature/M_d.sum()).reshape(n_feature)
    # M=torch.diag(M_d)
    if low_rank_yes_no==0:
        Cholesky_U_0[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]=M_d_in
        M0=Cholesky_U_0@Cholesky_U_0.T
    else:
        M0=M_rec@M_rec.T
        
    factor_for_diag=torch.trace(M0)/M_normalizer
    M=M0/factor_for_diag    
    L_M=graph_construction(feature_train, n_train, n_feature, M)
    metric_M_obj_M=torch.matmul(torch.matmul(label[b_ind].reshape(1,n_train),L_M),\
                                label[b_ind].reshape(n_train,1))
    tol_current=torch.norm(metric_M_obj_M-metric_M_obj)
    return M_d_in,M,tol_current,M_rec