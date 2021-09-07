from construct_H import construct_H
from lobpcg_first_eigen import lobpcg_first_eigen
from compute_scalars_scal import compute_scalars_scal
from gdpa_core import gdpa_core

def gdpa_full_process(label,\
                      n_train,\
                      n_test,\
                      b_ind,\
                      alpha,\
                      L,\
                      n_sample,\
                      rho,\
                      sw,\
                      scalee,\
                      y,\
                      z,\
                      dz_ind_plus,\
                      dz_ind_minus,\
                      lobpcg_fv,\
                      numiter,\
                      toler,\
                      db):

    alpha=sw*(y[-1].reshape(1)+z.sum())     
    
    initial_H=construct_H(n_sample,\
                L,\
                z,\
                y,\
                dz_ind_minus,\
                dz_ind_plus,\
                sw,\
                alpha)    
    
    evalue,evector=lobpcg_first_eigen(initial_H,\
                                      lobpcg_fv,\
                                      numiter,\
                                      toler)

    scaled_M,scaled_factors=\
        compute_scalars_scal(initial_H,n_sample,evector)    
        
    y0,z0,current_obj=gdpa_core(n_sample,\
                  n_train,\
                  n_test,\
                  scaled_M,\
                  scaled_factors,\
                  rho,\
                  db,\
                  L,\
                  b_ind,\
                  alpha,\
                  sw,\
                  dz_ind_plus,\
                  dz_ind_minus)
    
    return y0,z0,evector,alpha