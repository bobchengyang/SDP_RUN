from metric_M_diagonal import metric_M_diagonal
from graph_construction import graph_construction
import torch
from gdpa_full_process import gdpa_full_process

def M_LOBPCG_LP_no_network2(M_normalizer,optimizer_no_network,read_data,\
                b_ind,\
                db,\
                read_label,\
                n_feature,\
                M_d_0,\
                n_train,\
                metric_M_step,\
                n_sample,\
                n_test,\
                alpha,\
                rho,\
                sw,\
                scalee,\
                y,\
                z,\
                dz_ind_plus,\
                dz_ind_minus,\
                lobpcg_fv,\
                numiter,\
                toler,Q_mask,\
                LLE_C1_unroll,\
                LLE_mask,\
                l_factor,\
                sim_mask,disim_mask,\
                LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
                M_rec,\
                low_rank_yes_no,\
                LLE_yes_no):
    # tol=1e10
    # loop_number=0
    # while loop_number<300:
    #     loop_number+=1
    #     M_d_0,M,tol=metric_M_diagonal(read_data,\
    #                 b_ind,\
    #                 read_label,\
    #                 n_feature,\
    #                 M_d_0,\
    #                 n_train,\
    #                 metric_M_step) 
    tol=1e10
    loop_number=0
    # while loop_number<numiter_ep:
    while tol>1e-2 and loop_number<1e3:
        optimizer_no_network.zero_grad()
        # print(loop_number)
        loop_number+=1
        M_d_0,M,tol,M_rec=metric_M_diagonal(M_normalizer,read_data,\
                    b_ind,\
                    read_label,\
                    n_feature,\
                    M_d_0,\
                    n_train,\
                    metric_M_step,Q_mask,optimizer_no_network,\
                    M_rec,\
                    low_rank_yes_no) 
            
    # M=torch.diag(M_d_0.reshape(n_feature))   
    L=graph_construction(read_data, n_sample, n_feature, M)
    # L=graph_construction(read_data, n_sample, n_feature, torch.eye(n_feature))
    
    tril_idx=torch.tril_indices(n_train+n_test,n_train+n_test,-1)
    LLE_C_matrix0=torch.zeros(n_train+n_test,n_train+n_test)
    LLE_C_matrix0[tril_idx[0,LLE_mask],tril_idx[1,LLE_mask]]=\
        LLE_C1_unroll\
        +LLE_C1_unroll_delta*sim_mask[tril_idx[0,LLE_mask],tril_idx[1,LLE_mask]]\
        -LLE_C1_unroll_gamma*disim_mask[tril_idx[0,LLE_mask],tril_idx[1,LLE_mask]]
    LLE_C_matrix=LLE_C_matrix0+LLE_C_matrix0.T    
    D_LLE_C=torch.diag(LLE_C_matrix.sum(axis=1))
    L_LLE_C=D_LLE_C-LLE_C_matrix
    
    if LLE_yes_no==0:
        cL=torch.multiply(L,scalee)
    else:
        cL=torch.multiply(L+l_factor*L_LLE_C,scalee)

    sccc=torch.trace(cL)
    y0=y*sccc
    z0=z*sccc

    # LOBPCG+LP
    y1,z1,lobpcg_fv1,alpha1=gdpa_full_process(read_data,\
                          n_train,\
                          n_test,\
                          b_ind,\
                          alpha,\
                          cL,\
                          n_sample,\
                          rho,\
                          sw,\
                          scalee,\
                          y0,\
                          z0,\
                          dz_ind_plus,\
                          dz_ind_minus,\
                          lobpcg_fv,\
                          numiter,\
                          toler,\
                          db)
    y2,z2,lobpcg_fv2,alpha2=gdpa_full_process(read_data,\
                          n_train,\
                          n_test,\
                          b_ind,\
                          alpha1,\
                          cL,\
                          n_sample,\
                          rho,\
                          sw,\
                          scalee,\
                          y1,\
                          z1,\
                          dz_ind_plus,\
                          dz_ind_minus,\
                          lobpcg_fv1,\
                          numiter,\
                          toler,\
                          db)
    # y3,z3,lobpcg_fv3,alpha3=gdpa_full_process(read_data,\
    #                       n_train,\
    #                       n_test,\
    #                       b_ind,\
    #                       alpha2,\
    #                       cL,\
    #                       n_sample,\
    #                       rho,\
    #                       sw,\
    #                       scalee,\
    #                       y2,\
    #                       z2,\
    #                       dz_ind_plus,\
    #                       dz_ind_minus,\
    #                       lobpcg_fv2,\
    #                       numiter,\
    #                       toler,\
    #                       db)
    # y4,z4,lobpcg_fv4,alpha4=gdpa_full_process(read_data,\
    #                       n_train,\
    #                       n_test,\
    #                       b_ind,\
    #                       alpha3,\
    #                       cL,\
    #                       n_sample,\
    #                       rho,\
    #                       sw,\
    #                       scalee,\
    #                       y3,\
    #                       z3,\
    #                       dz_ind_plus,\
    #                       dz_ind_minus,\
    #                       lobpcg_fv3,\
    #                       numiter,\
    #                       toler,\
    #                       db)       
    # y5,z5,lobpcg_fv5,alpha5=gdpa_full_process(read_data,\
    #                       n_train,\
    #                       n_test,\
    #                       b_ind,\
    #                       alpha4,\
    #                       cL,\
    #                       n_sample,\
    #                       rho,\
    #                       sw,\
    #                       scalee,\
    #                       y4,\
    #                       z4,\
    #                       dz_ind_plus,\
    #                       dz_ind_minus,\
    #                       lobpcg_fv4,\
    #                       numiter,\
    #                       toler,\
    #                       db)        
    return y2,z2,lobpcg_fv2,M_d_0,cL,read_data,alpha2 