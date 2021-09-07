import torch

def construct_H(n_sample,\
                cL,\
                z,\
                y,\
                dz_ind_minus,\
                dz_ind_plus,\
                sw,\
                alpha):
    right_np1=torch.zeros(n_sample)
    right_np1.scatter_(0,dz_ind_minus,z[dz_ind_minus])
    
    right_np2=torch.zeros(n_sample)
    right_np2.scatter_(0,dz_ind_plus,z[dz_ind_plus])
    
    right_np12=torch.cat((right_np1.reshape((n_sample,1)),right_np2.reshape((n_sample,1))),dim=1)
    bottom_np12=torch.cat((right_np1.reshape((1,n_sample)),right_np2.reshape((1,n_sample))),dim=0)
    
    node_np1=sw*(y[n_sample]+\
                 z.sum())-\
                 z[dz_ind_minus].sum()-alpha
    node_np2=(1-sw)*(y[n_sample]+\
                 z.sum())-\
                 z[dz_ind_plus].sum()+alpha
    corner_np12=torch.diag(torch.cat((node_np1,node_np2)))
    
    cLy=cL+torch.diag(y[0:n_sample])
    cL_right=torch.cat((cLy,right_np12),dim=1)
    bottom_np12_corner_np12=torch.cat((bottom_np12,corner_np12),dim=1)
    
    initial_H=torch.cat((cL_right,bottom_np12_corner_np12),dim=0)
    return initial_H