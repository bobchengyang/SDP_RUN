from low_rank_approx import low_rank_approx
from torch.autograd import Variable

def low_rank_approx_list(n_feature,\
                    Q_mask,\
                    M_d_0,\
                    M_d_0_,\
                    M_d_0__,\
                    low_rank_k):    
      
    M_rec1=low_rank_approx(n_feature,Q_mask,M_d_0,low_rank_k)
    M_rec2=low_rank_approx(n_feature,Q_mask,M_d_0_,low_rank_k)
    M_rec3=low_rank_approx(n_feature,Q_mask,M_d_0__,low_rank_k)
    
    M_rec1 = Variable(M_rec1, requires_grad=True) # set the initial low-rank matrix
    M_rec2 = Variable(M_rec2, requires_grad=True) # set the initial low-rank matrix
    M_rec3 = Variable(M_rec3, requires_grad=True) # set the initial low-rank matrix
    return M_rec1,M_rec2,M_rec3