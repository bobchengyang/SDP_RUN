from torch.autograd import Variable

def LLE_relu_set(LLE_C_relu,\
                 LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                 LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
                 LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3):
    LLE_C1_unroll=Variable(LLE_C_relu(LLE_C1_unroll).detach(),requires_grad=True)
    LLE_C2_unroll=Variable(LLE_C_relu(LLE_C2_unroll).detach(),requires_grad=True)
    LLE_C3_unroll=Variable(LLE_C_relu(LLE_C3_unroll).detach(),requires_grad=True)
    LLE_C1_unroll_delta=Variable(LLE_C_relu(LLE_C1_unroll_delta).detach(),requires_grad=True)
    LLE_C2_unroll_delta=Variable(LLE_C_relu(LLE_C2_unroll_delta).detach(),requires_grad=True)
    LLE_C3_unroll_delta=Variable(LLE_C_relu(LLE_C3_unroll_delta).detach(),requires_grad=True)
    LLE_C1_unroll_gamma=Variable(LLE_C_relu(LLE_C1_unroll_gamma).detach(),requires_grad=True)
    LLE_C2_unroll_gamma=Variable(LLE_C_relu(LLE_C2_unroll_gamma).detach(),requires_grad=True)
    LLE_C3_unroll_gamma=Variable(LLE_C_relu(LLE_C3_unroll_gamma).detach(),requires_grad=True)
    l_factor_1=Variable(LLE_C_relu(l_factor_1).detach(),requires_grad=True)
    l_factor_2=Variable(LLE_C_relu(l_factor_2).detach(),requires_grad=True)
    l_factor_3=Variable(LLE_C_relu(l_factor_3).detach(),requires_grad=True)
    return LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
           LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
           LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3