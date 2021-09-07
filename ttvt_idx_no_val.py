import torch

def ttvt_idx_no_val(n_sample,train_all_n,train_net_n,val_n):
    b_ind=torch.arange(0,round(train_all_n*n_sample)) # ~50% training_all sample
    b_ind_train_net=b_ind.shape[0]+torch.arange(0,round(train_net_n*n_sample)) # ~20% training_net sample
    b_ind_val=b_ind_train_net
    n_train=b_ind.shape[0] # number of training_all sample
    n_train_net=b_ind_train_net.shape[0] # number of training_net sample
    n_val=n_train_net
    n_test=n_sample-n_train-n_train_net # number of test sample
    b_ind_test=n_train+n_train_net+torch.arange(0,n_test) # 20% test sample
    return n_train,n_train_net,n_val,n_test,b_ind,b_ind_train_net,b_ind_val,b_ind_test