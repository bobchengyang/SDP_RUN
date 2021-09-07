import torch
from z_score_normalize import z_score_normalize
import numpy as np
from lobpcg_first_eigen import lobpcg_first_eigen
from diag_idx import diag_idx
from scale_01 import scale_01

def dataloader_normalized_standardization_model_based(feature,\
                                          label,\
                                          n_sample,\
                                          n_feature,\
                                          b_ind,\
                                          b_ind_train_net,\
                                          b_ind_val,\
                                          b_ind_test,\
                                          rsrng,\
                                          std_or_01,\
                                          sparsity_threshold01):
    torch.random.manual_seed(rsrng) 
    reorderx=torch.randperm(n_sample)
    feature=feature[reorderx,:]
    label=label[reorderx]
    
    b_ind_train_model_based=torch.cat((b_ind,b_ind_train_net))
    
    feature_train=feature[b_ind_train_model_based,:]
    if std_or_01==1:
        feature_train=z_score_normalize(feature_train,b_ind_train_model_based)
    else:
        feature_train=scale_01(feature_train,n_feature)
    feature_train_NaN=np.isnan(feature_train)
    column_to_remove=[]
    for ii in range(n_feature):
        if feature_train_NaN[:,ii].any()==1: # NaN exists
            column_to_remove.append(ii)
    
    feature_train_net=feature[b_ind_train_net,:]
    if std_or_01==1:
        feature_train_net=z_score_normalize(feature_train_net,b_ind_train_net)
    else:
        feature_train_net=scale_01(feature_train_net,n_feature)
    feature_train_net_NaN=np.isnan(feature_train_net)
    for ii in range(n_feature):
        if feature_train_net_NaN[:,ii].any()==1: # NaN exists
            column_to_remove.append(ii)    
            
    feature_val=feature[b_ind_val,:]
    if std_or_01==1:
        feature_val=z_score_normalize(feature_val,b_ind_val)
    else:
        feature_val=scale_01(feature_val,n_feature)
    feature_val_NaN=np.isnan(feature_val)
    for ii in range(n_feature):
        if feature_val_NaN[:,ii].any()==1: # NaN exists
            column_to_remove.append(ii)    
            
    feature_test=feature[b_ind_test,:]
    if std_or_01==1:
        feature_test=z_score_normalize(feature_test,b_ind_test)
    else:
        feature_test=scale_01(feature_test,n_feature)
    feature_test_NaN=np.isnan(feature_test)
    for ii in range(n_feature):
        if feature_test_NaN[:,ii].any()==1: # NaN exists
            column_to_remove.append(ii)     
    
    feature_train=np.delete(feature_train, column_to_remove, axis=1)
    feature_train_net=np.delete(feature_train_net, column_to_remove, axis=1)
    feature_val=np.delete(feature_val, column_to_remove, axis=1)
    feature_test=np.delete(feature_test, column_to_remove, axis=1)
    
    n_feature=feature_train.shape[1]    
    
    feature_train_cov=np.cov(feature_train.T)
    
    torch.random.manual_seed(0)
    lfv=torch.randn(n_feature,1)
    if n_feature>2:
        d,v=lobpcg_first_eigen(torch.from_numpy(feature_train_cov).float(),\
                               lfv,\
                               200,\
                               1e-4)
        if d<=0:
            feature_train_cov_Q=np.linalg.cholesky(feature_train_cov+(-d.detach().numpy()+1e-5)*np.eye(n_feature))
        else:
            feature_train_cov_Q=np.linalg.cholesky(feature_train_cov+(d.detach().numpy()+1e-5)*np.eye(n_feature))
    else:
        d,v=np.linalg.eig(feature_train_cov)
        if min(d)<=0:
            feature_train_cov_Q=np.linalg.cholesky(feature_train_cov+(-min(d)+1e-5)*np.eye(n_feature))
        else:
            feature_train_cov_Q=np.linalg.cholesky(feature_train_cov)
            
    # feature_train_cov_mask=np.abs(feature_train_cov) < np.diag(feature_train_cov).mean()*0.95
    feature_train_cov_mask=np.abs(feature_train_cov_Q) < np.diag(feature_train_cov_Q).mean()*sparsity_threshold01
    print(np.count_nonzero(~feature_train_cov_mask))
    diagidx=diag_idx(n_feature)
    feature_train_cov_mask[diagidx[0],diagidx[1]]=False
    print(np.count_nonzero(~feature_train_cov_mask))
    tril_idx=torch.tril_indices(n_feature,n_feature)
    Q_mask=~feature_train_cov_mask[tril_idx[0],tril_idx[1]]
    nvariables=np.count_nonzero(Q_mask)
    initial_Q_vec=feature_train_cov_Q[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]
    print(nvariables)
    
    feature_train_torch=torch.from_numpy(feature_train)
    feature_train_net_torch=torch.from_numpy(feature_train_net)
    feature_val_torch=torch.from_numpy(feature_val)
    feature_test_torch=torch.from_numpy(feature_test)

    # feature_reform_train_net=torch.cat((feature_train_torch,feature_train_net_torch),dim=0)
    feature_reform_train_net=feature_train_torch
    feature_reform_val=torch.cat((feature_train_torch,feature_val_torch),dim=0)
    feature_reform_test=torch.cat((feature_train_torch,feature_test_torch),dim=0)
    
    label_torch=torch.from_numpy(label)
    # label_train_net=label_torch[torch.cat((b_ind,b_ind_train_net))]
    label_train_net=label_torch[b_ind_train_model_based]
    # label_val=label_torch[torch.cat((b_ind,b_ind_val))]
    # label_test=label_torch[torch.cat((b_ind,b_ind_test))]
    label_val=label_torch[torch.cat((b_ind_train_model_based,b_ind_val))]
    label_test=label_torch[torch.cat((b_ind_train_model_based,b_ind_test))]    
    return feature_reform_train_net,feature_reform_val,feature_reform_test,\
           n_feature,\
           label_train_net,label_val,label_test,\
           Q_mask,nvariables,initial_Q_vec