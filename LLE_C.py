import torch
import torch.nn as nn
from torch.autograd import Variable

def LLE_C(feature,\
          b_ind,\
          lalel,\
          n_feature,\
          LLE_C_vec,\
          n_train,\
          metric_M_step,\
          LLE_st,\
          optimizer_LLE):
    
    tril_idx=torch.tril_indices(n_train,n_train,-1)
    LLE_C_matrix0=torch.zeros(n_train,n_train)
    LLE_C_matrix0[tril_idx[0],tril_idx[1]]=LLE_C_vec
    LLE_C_matrix=LLE_C_matrix0+LLE_C_matrix0.T
    
    feature_train=feature[b_ind,:]
    LLE_obj=torch.linalg.norm(feature_train-LLE_C_matrix@feature_train)**2
    l1_before=torch.linalg.norm(LLE_C_matrix,ord=1)
    LLE_obj.backward()
    optimizer_LLE.step()
    
    LLE_C_relu=nn.ReLU()
    LLE_C_vec=LLE_C_relu(LLE_C_vec-LLE_st)

    LLE_C_matrix0[tril_idx[0],tril_idx[1]]=LLE_C_vec
    LLE_C_matrix_C=LLE_C_matrix0+LLE_C_matrix0.T
    
    LLE_obj_C=torch.linalg.norm(feature_train-LLE_C_matrix_C@feature_train)**2
    l1_after=torch.linalg.norm(LLE_C_matrix_C,ord=1)
    tol_current=torch.norm(LLE_obj_C+l1_after-(LLE_obj+l1_before))
    return Variable(LLE_C_vec.detach(), requires_grad=True),tol_current