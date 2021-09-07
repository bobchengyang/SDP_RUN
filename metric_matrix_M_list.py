from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim
from metric_M_diagonal import metric_M_diagonal
from low_rank_approx_list import low_rank_approx_list

def metric_matrix_M_list(M_normalizer,random_number,\
                         initial_Q_vec,\
                         lr,\
                         read_data_train_net,\
                         b_ind,\
                         read_label_train_net,\
                         n_feature,\
                         n_train,\
                         metric_M_step,\
                         Q_mask,\
                         low_rank_k,\
                         low_rank_yes_no):
    torch.random.manual_seed(random_number) # set the random seeds
    M_d_0 = Variable(torch.from_numpy(np.float32(initial_Q_vec)), requires_grad=True) # set the initial lower-triangular matrix
    torch.random.manual_seed(random_number) # set the random seeds
    M_d_0_ = Variable(torch.from_numpy(np.float32(initial_Q_vec)), requires_grad=True) # set the initial lower-triangular matrix
    torch.random.manual_seed(random_number) # set the random seeds
    M_d_0__ = Variable(torch.from_numpy(np.float32(initial_Q_vec)), requires_grad=True) # set the initial lower-triangular matrix
    
    M_rec1,\
    M_rec2,\
    M_rec3=low_rank_approx_list(n_feature,\
            Q_mask,\
            M_d_0,\
            M_d_0_,\
            M_d_0__,\
            low_rank_k)
                
    tol=1e10
    loop_number=0
    # while loop_number<numiter_ep:
    optimizer_M=optim.Adam([M_d_0],lr=lr)    
    while tol>1e-2 and loop_number<1e3:
        optimizer_M.zero_grad()
        loop_number+=1
        M_d_0,Mm,tol,M_rec1=metric_M_diagonal(M_normalizer,read_data_train_net,\
                    b_ind,\
                    read_label_train_net,\
                    n_feature,\
                    M_d_0,\
                    n_train,\
                    metric_M_step,Q_mask,optimizer_M,\
                    M_rec1,\
                    low_rank_yes_no) 
          

    tol=1e10
    loop_number=0
    # while loop_number<numiter_ep:
    optimizer_M=optim.Adam([M_d_0_],lr=lr)    
    while tol>1e-2 and loop_number<1e3:
        optimizer_M.zero_grad()
        # print(loop_number)
        loop_number+=1
        M_d_0_,Mm,tol,M_rec2=metric_M_diagonal(M_normalizer,read_data_train_net,\
                    b_ind,\
                    read_label_train_net,\
                    n_feature,\
                    M_d_0_,\
                    n_train,\
                    metric_M_step,Q_mask,optimizer_M,\
                    M_rec2,\
                    low_rank_yes_no) 
        
    tol=1e10
    loop_number=0
    # while loop_number<numiter_ep:
    optimizer_M=optim.Adam([M_d_0__],lr=lr)      
    while tol>1e-2 and loop_number<1e3:
        optimizer_M.zero_grad()
        # print(loop_number)
        loop_number+=1
        M_d_0__,Mm,tol,M_rec3=metric_M_diagonal(M_normalizer,read_data_train_net,\
                    b_ind,\
                    read_label_train_net,\
                    n_feature,\
                    M_d_0__,\
                    n_train,\
                    metric_M_step,Q_mask,optimizer_M,\
                    M_rec3,\
                    low_rank_yes_no) 
    return M_d_0,\
           M_d_0_,\
           M_d_0__,\
           M_rec1,\
           M_rec2,\
           M_rec3