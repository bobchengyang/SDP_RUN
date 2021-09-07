import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from LLE_C import LLE_C
def get_LLE_set(random_number,\
                n_train,\
                n_train_net,\
                read_data_train_net,\
                b_ind,\
                b_ind_train_net,\
                read_label_train_net,\
                n_feature,\
                metric_M_step,\
                LLE_st,\
                lr):
    torch.random.manual_seed(random_number)
    LLE_Cv=Variable(torch.ones(int((n_train+n_train_net)*((n_train+n_train_net)-1)/2)), requires_grad=True)
    
    tol=1e10
    loop_number=0
    # while loop_number<numiter_ep:   
    while tol>1e-2 and loop_number<1e3:
        optimizer_LLE=optim.Adam([LLE_Cv],lr=lr)  
        optimizer_LLE.zero_grad()
        # print(loop_number)
        loop_number+=1
        LLE_Cv,tol=LLE_C(read_data_train_net,\
                    torch.cat((b_ind,b_ind_train_net)),\
                    read_label_train_net,\
                    n_feature,\
                    LLE_Cv,\
                    n_train+n_train_net,\
                    metric_M_step,\
                    LLE_st,\
                    optimizer_LLE)
    
    LLE_mask=LLE_Cv!=0
    nvar_LLE=np.count_nonzero(LLE_mask)
    torch.random.manual_seed(0)
    LLE_C_unroll=Variable(LLE_Cv[LLE_mask], requires_grad=True)
    LLE_C_unroll_delta=Variable(torch.zeros(1),requires_grad=True)
    LLE_C_unroll_gamma=Variable(torch.zeros(1),requires_grad=True)
    l_factor=Variable(torch.zeros(1),requires_grad=True)
    return LLE_mask,\
           LLE_C_unroll,\
           LLE_C_unroll_delta,\
           LLE_C_unroll_gamma,\
           l_factor