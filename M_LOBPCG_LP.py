import torch
from graph_construction import graph_construction
from gdpa_full_process import gdpa_full_process

def M_LOBPCG_LP(M_normalizer,read_data,\
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
                toler,sccc_i,Q_mask,LLE_C1_unroll,LLE_mask,l_factor,sim_mask,disim_mask,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
                M_rec,low_rank_yes_no,\
                LLE_yes_no):
    # M diagonal unrolling (3 layers)
    # M_d_out1=metric_M_diagonal(read_data,\
    #                     b_ind,\
    #                     read_label,\
    #                     n_feature,\
    #                     M_d_in,\
    #                     n_train,\
    #                     metric_M_step)    
    # print(M_d_out1)   
    # M_d_out2=metric_M_diagonal(read_data,\
    #                     b_ind,\
    #                     read_label,\
    #                     n_feature,\
    #                     M_d_out1,\
    #                     n_train,\
    #                     metric_M_step)    
    # print(M_d_out2)                
    # M_d_out3=metric_M_diagonal(read_data,\
    #                     b_ind,\
    #                     read_label,\
    #                     n_feature,\
    #                     M_d_out2,\
    #                     n_train,\
    #                     metric_M_step)    
    # print(M_d_out3)                
    
    # M_d_out1=metric_M_diagonal_cvxpylayer(read_data,\
    #                               b_ind,\
    #                               read_label,\
    #                               n_feature,\
    #                               n_train)
    
    # read_data_new_train=net1(read_data[b_ind,:]) # net() requires float32 not float64!
    # read_data_new_test=net1(read_data[n_train:n_sample,:]) # net() requires float32 not float64!
    # read_data_new=torch.cat((read_data_new_train,read_data_new_test),dim=0)
    
    # read_data_new_normalized=dataloader_normalized_standardization_tensor(read_data_new,\
    #                         n_train,\
    #                         n_sample,\
    #                         b_ind)       

    # tol=1e10
    # loop_number=0
    # while loop_number<100:
    #     loop_number+=1
    #     # print(loop_number)
    #     M_d_0,tol=metric_M_diagonal(read_data_new.detach(),\
    #                 b_ind,\
    #                 read_label,\
    #                 n_feature,\
    #                 M_d_0,\
    #                 n_train,\
    #                 metric_M_step) 
            
    # M_d_in0=net2(M_d_0)
    # M_d_in0_=M_d_in0-M_d_in0.min()
    # M_d_10=M_d_in0_/M_d_in0_.max()
    # M_d_1=M_d_10*n_feature/M_d_10.sum()
    # M_d_1=M_d_in0*n_feature
    # M=torch.diag(M_d_1.reshape(n_feature))   
    if low_rank_yes_no==0:
        tril_idx=torch.tril_indices(n_feature,n_feature)
        Cholesky_U_0=torch.zeros(n_feature,n_feature)
        Cholesky_U_0[tril_idx[0,Q_mask],tril_idx[1,Q_mask]]=M_d_0
        M0=Cholesky_U_0@Cholesky_U_0.T
    else:
        M0=M_rec@M_rec.T
    
    factor_for_diag=torch.trace(M0)/M_normalizer
    M=M0/factor_for_diag
    
    L=graph_construction(read_data, n_sample, n_feature, M)
    
    tril_idx=torch.tril_indices(n_sample,n_sample,-1)
    LLE_C_matrix0=torch.zeros(n_sample,n_sample)
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
    
    # if torch.norm(M_d_0-torch.ones(n_feature))==0:
    if sccc_i==0:
        sccc=torch.trace(cL)
        y0=y*sccc
        z0=z*sccc
    else:
        y0=y
        z0=z

    # LOBPCG+LP unrolling (3 layers)
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
                          toler,db)
    # y2,z2,lobpcg_fv2=gdpa_full_process(read_data,\
    #                       n_train,\
    #                       n_test,\
    #                       b_ind,\
    #                       alpha,\
    #                       cL,\
    #                       n_sample,\
    #                       rho,\
    #                       sw,\
    #                       scalee,\
    #                       y1,\
    #                       z1,\
    #                       dz_ind_plus,\
    #                       dz_ind_minus,\
    #                       lobpcg_fv1,\
    #                       numiter,\
    #                       toler,db)
    # y3,z3,lobpcg_fv3=gdpa_full_process(read_data,\
    #                       n_train,\
    #                       n_test,\
    #                       b_ind,\
    #                       alpha,\
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
    #                       toler,db)
    return y1,z1,lobpcg_fv1,M,cL,read_data,alpha1,LLE_C_matrix