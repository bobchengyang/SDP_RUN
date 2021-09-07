import torch
from diag_idx import diag_idx

def LLE_delta_gamma_mask(n_train,\
                         n_train_net,\
                         read_label_train_net,\
                         b_ind):
    sim_disim_mask=torch.zeros(n_train+n_train_net,n_train+n_train_net)
    sim_disim_block=read_label_train_net[b_ind].reshape((read_label_train_net[b_ind].shape[0],1))*\
        read_label_train_net[b_ind].reshape((1,read_label_train_net[b_ind].shape[0]))
    sim_disim_mask[0:n_train,0:n_train]=sim_disim_block
    sim_mask=sim_disim_mask==1
    disim_mask=sim_disim_mask==-1
    diagidx=diag_idx(n_train)
    sim_mask[diagidx[0],diagidx[1]]=False
    return sim_mask,disim_mask