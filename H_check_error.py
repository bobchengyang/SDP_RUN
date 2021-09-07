import torch
from lobpcg_first_eigen import lobpcg_first_eigen

def H_check_error(label,\
                  y,\
                  z,\
                  n_sample,\
                  cL,\
                  b_ind,\
                  numiter,\
                  toler,\
                  b_ind_test,\
                  n_train):
    
    right_n_sample=torch.zeros(n_sample)
    right_n_sample.scatter_(0,b_ind,z)
    
    cL_right=torch.cat((cL,right_n_sample.reshape(n_sample,1)),dim=1)
    bottom_np1=torch.cat((right_n_sample,torch.tensor([0])),dim=0).reshape((1,n_sample+1))
    
    original_H_o=torch.cat((cL_right,bottom_np1),dim=0)
    original_H=original_H_o+torch.diag(y)
    
    torch.random.manual_seed(0)
    lobpcg_fv=torch.randn(n_sample+1,1)
            
    evalue,evector=lobpcg_first_eigen(original_H,\
                                      lobpcg_fv,\
                                      numiter,\
                                      toler)
    
    b_ind_n_sample=torch.cat((b_ind,torch.tensor([n_sample])),dim=0)
    print(f"energy in b_ind n_sample+1: {torch.norm(evector[b_ind_n_sample])}")
    print(f"energy in z: {torch.norm(evector[n_train:n_sample])}")
    
    # x_val=torch.sign(label[b_ind[0]])*\
    #       torch.sign(evector[0])*\
    #       torch.sign(evector[0:n_sample]).reshape((n_sample))
    # x_val=torch.nn.functional.softsign(label[b_ind[0]])*\
    #       torch.nn.functional.softsign(evector[0])*\
    #       torch.nn.functional.softsign(evector[0:n_sample]).reshape((n_sample)) 
    x_val0=label[b_ind[0]]*evector[0]*evector[0:n_sample].reshape(n_sample)
    # x_val00=x_val0.detach()
    # x_val=torch.nn.functional.softsign(x_val0)
    x_val=x_val0/x_val0.detach().abs()
    # x_val=x_val0/x_val0.abs()
    err_count=torch.sum(torch.abs(torch.sign(x_val[b_ind_test])-label[b_ind_test]))/2    
    return err_count,x_val