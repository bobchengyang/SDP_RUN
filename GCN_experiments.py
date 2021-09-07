import torch
import numpy as np
from torch.autograd import Variable
from GCNlayer import GCNlayer
import torch.nn as nn
import torch.optim as optim
from graph_construction import graph_construction
import torch.nn.functional as F

def GCN_experiments(random_number,\
                    initial_Q_vec,\
                    n_feature,\
                    nneuron,\
                    read_label_train_net,\
                    read_label_val,\
                    read_label_test,\
                    nvariables,\
                    total_run_i_net,\
                    numiter_nn,\
                    final_results,\
                    Q_mask,\
                    read_data_train_net,\
                    n_train,\
                    n_train_net,\
                    read_data_val,\
                    n_val,\
                    read_data_test,\
                    b_ind,\
                    n_test,\
                    results,\
                    K_i,\
                    rsrng,\
                    num_run0):
    torch.random.manual_seed(random_number) # set the random seeds
    M_d_0_gcn = Variable(torch.from_numpy(np.float32(initial_Q_vec)), requires_grad=True) # set the initial lower triangular matrix
    netnn = nn.Sequential(GCNlayer(n_feature,2,nneuron,M_d_0_gcn)) # build the black-box neural network
    optimizer_nn=optim.Adam(netnn.parameters(),lr=1e-2) # set the optimizer for the black-box neural network
    
    read_label_train_net_mlp=read_label_train_net.clone() # get the label -1 to 0
    read_label_train_net_mlp[read_label_train_net_mlp==-1]=0 # get the label -1 to 0
    read_label_val_mlp=read_label_val.clone() # get the label -1 to 0
    read_label_val_mlp[read_label_val_mlp==-1]=0 # get the label -1 to 0
    read_label_test_mlp=read_label_test.clone() # get the label -1 to 0
    read_label_test_mlp[read_label_test_mlp==-1]=0 # get the label -1 to 0
    
    loss_mlp_log=[]
    error_mlp_train_net_log=[]
    error_mlp_val_log=[]
    final_results[total_run_i_net,0]=nvariables+n_feature*nneuron+nneuron*nneuron+nneuron*2+nneuron+nneuron+2
    print(f"Total number of parameters to be learnt: {nvariables+n_feature*nneuron+nneuron*nneuron+nneuron*2+nneuron+nneuron+2}")
    
    for _ in range(numiter_nn):
        optimizer_nn.zero_grad()
        
        tril_idx=torch.tril_indices(n_feature,n_feature)
        Cholesky_U_0=torch.zeros(n_feature,n_feature)
        Cholesky_U_0[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]=M_d_0_gcn
        M0=Cholesky_U_0@Cholesky_U_0.T
        factor_for_diag=torch.trace(M0)/n_feature
        M_gcn=M0/factor_for_diag

        read_data_train_net_adj=graph_construction(read_data_train_net, n_train+n_train_net, n_feature, M_gcn)
        read_data_train_net_data=[]
        read_data_train_net_data.append(read_data_train_net)
        read_data_train_net_data.append(read_data_train_net_adj)
        predicted_mlp=netnn(read_data_train_net_data)
        predicted_mlp_softmax=F.softmax(predicted_mlp,dim=1)
        predicted_mlp_log_softmax=F.log_softmax(predicted_mlp)
        loss_mlp = nn.CrossEntropyLoss()(predicted_mlp,read_label_train_net_mlp.type(torch.LongTensor))
        prediction_error=(predicted_mlp_log_softmax.max(dim=1).indices-\
            read_label_train_net_mlp.type(torch.LongTensor)).count_nonzero()
        if _ % 50 == 0:
            print(f"loss_mlp label train_net: {loss_mlp.item()}")
            print(f"prediction error train_net: {prediction_error}")
        loss_mlp.backward()
        optimizer_nn.step() # Does the update          
        read_data_val_adj=graph_construction(read_data_val, n_train+n_val, n_feature, M_gcn)            
        read_data_val_data=[]
        read_data_val_data.append(read_data_val)
        read_data_val_data.append(read_data_val_adj)
        predicted_mlp_val=netnn(read_data_val_data)
        predicted_mlp_softmax_val=F.softmax(predicted_mlp_val,dim=1)
        predicted_mlp_log_softmax_val=F.log_softmax(predicted_mlp_val)
        prediction_error_val=(predicted_mlp_log_softmax_val.max(dim=1).indices-\
        read_label_val_mlp.type(torch.LongTensor)).count_nonzero()
        if _ % 50 == 0:
            print(f"prediction error val: {prediction_error_val}")     
        
        loss_mlp_log.append(loss_mlp.item())
        error_mlp_train_net_log.append(prediction_error)
        error_mlp_val_log.append(prediction_error_val)
    
    read_data_test_adj=graph_construction(read_data_test[b_ind.shape[0]+torch.arange(0,n_test),:], n_test, n_feature, M_gcn)            
    read_data_test_data=[]
    read_data_test_data.append(read_data_test[b_ind.shape[0]+torch.arange(0,n_test),:])
    read_data_test_data.append(read_data_test_adj)
    predicted_mlp_test=netnn(read_data_test_data)
    predicted_mlp_softmax_test=F.softmax(predicted_mlp_test,dim=1)
    predicted_mlp_log_softmax_test=F.log_softmax(predicted_mlp_test)
    prediction_error_test=(predicted_mlp_log_softmax_test.max(dim=1).indices-\
    read_label_test_mlp[b_ind.shape[0]+torch.arange(0,n_test)].type(torch.LongTensor)).count_nonzero()
    results[rsrng+(K_i-1)*num_run0,0]=prediction_error_test
    print(f"error_count black-box: {prediction_error_test}")
    print("=========================================")     
    return results