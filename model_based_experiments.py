import torch
from dataloader_normalized_standardization_model_based import dataloader_normalized_standardization_model_based
from LLE_delta_gamma_mask import LLE_delta_gamma_mask
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from LLE_C import LLE_C
from M_LOBPCG_LP_no_network import M_LOBPCG_LP_no_network
from H_check_error import H_check_error
from M_LOBPCG_LP_no_network3 import M_LOBPCG_LP_no_network3
from low_rank_approx_list import low_rank_approx_list
from M_LOBPCG_LP_no_network2 import M_LOBPCG_LP_no_network2

def model_based_experiments(M_normalizer_i,read_data_np_i_feature,\
                            read_data_np_i_label,\
                            n_sample,\
                            n_feature,\
                            b_ind,\
                            b_ind_train_net,\
                            b_ind_val,\
                            b_ind_test,\
                            rsrng,\
                            std_or_01,\
                            n_train,\
                            n_train_net,\
                            n_test,\
                            lr,\
                            final_results,\
                            total_run_i_net,\
                            metric_M_step,\
                            alpha,\
                            rho,\
                            sw,\
                            scalee,\
                            numiter,\
                            toler,\
                            results,\
                            K_i,\
                            num_run0,\
                            sparsity_threshold01,\
                            layer_i,\
                            low_rank_yes_no,\
                            low_rank_k,\
                            LLE_yes_no):
    
    read_data_train_net,read_data_val,read_data_test,\
    n_feature,\
    read_label_train_net,read_label_val,read_label_test,\
    Q_mask,nvariables,initial_Q_vec=\
        dataloader_normalized_standardization_model_based(read_data_np_i_feature,\
                                    read_data_np_i_label,\
                                    n_sample,\
                                    n_feature,\
                                    b_ind,\
                                    b_ind_train_net,\
                                    b_ind_val,\
                                    b_ind_test,\
                                    rsrng,\
                                    std_or_01,\
                                    sparsity_threshold01)            
                
    if M_normalizer_i==1:
        M_normalizer=n_feature
    else:
        M_normalizer=1
    
    sim_mask,disim_mask=LLE_delta_gamma_mask(n_train+n_train_net,n_test,read_label_train_net,torch.arange(0,n_train+n_train_net))

    bbbbb = torch.cat((b_ind,b_ind_train_net))
    db = 2*read_label_test[bbbbb] # training_all labels x 2
    dz_plus_idx = db < 0
    dz_minus_idx = db > 0
    dz_ind_plus = bbbbb[dz_plus_idx]
    dz_ind_minus = bbbbb[dz_minus_idx] 
    
    y_known=torch.ones(n_train+n_train_net)
    # y_unknown_train_net=torch.ones(n_train_net)
    # y_unknown_val=torch.ones(n_val)
    y_unknown_test=torch.ones(n_test)
    y_np1=torch.Tensor([n_train+n_train_net])
    # y_train_net=torch.cat((torch.cat((y_known,y_unknown_train_net),dim=0),y_np1),dim=0)
    # y_val=torch.cat((torch.cat((y_known,y_unknown_val),dim=0),y_np1),dim=0)
    y_test=torch.cat((torch.cat((y_known,y_unknown_test),dim=0),y_np1),dim=0)
    z=-(db/2)  
    
    torch.random.manual_seed(0)
    lobpcg_fv_test=torch.randn(n_train+n_train_net+n_test+2,1)
    
    torch.random.manual_seed(0)
    M_d_0 = Variable(torch.from_numpy(np.float32(initial_Q_vec)), requires_grad=True)
    
    M_rec,\
    M_rec,\
    M_rec=low_rank_approx_list(n_feature,\
            Q_mask,\
            M_d_0,\
            M_d_0,\
            M_d_0,\
            low_rank_k)
                
    # M_d_0 = Variable(torch.randn(int(nvariables)), requires_grad=True)
    if low_rank_yes_no==0:
         optimizer_no_network=optim.Adam([M_d_0],lr=lr)
    else:
         optimizer_no_network=optim.Adam([M_rec],lr=lr)

    #===============================================================================
    torch.random.manual_seed(0)
    LLE_C1=Variable(torch.randn(int((n_train+n_train_net+n_test)*((n_train+n_train_net+n_test)-1)/2)), requires_grad=True)
    tol=1e10
    LLE_st=1e-3
    loop_number=0
    # while loop_number<numiter_ep:   
    while tol>1e-2 and loop_number<1e3:
        optimizer_LLE=optim.Adam([LLE_C1],lr=lr)  
        optimizer_LLE.zero_grad()
        # print(loop_number)
        loop_number+=1
        LLE_C1,tol=LLE_C(read_data_test,\
                    torch.arange(n_train+n_train_net+n_test),\
                    read_label_test,\
                    n_feature,\
                    LLE_C1,\
                    n_train+n_train_net+n_test,\
                    metric_M_step,\
                    LLE_st,\
                    optimizer_LLE)
    
    LLE_mask_1=LLE_C1!=0
    nvar_LLE_1=np.count_nonzero(LLE_mask_1)
    torch.random.manual_seed(0)
    LLE_C1_unroll=Variable(LLE_C1[LLE_mask_1], requires_grad=True)
    LLE_C1_unroll_delta=1e-9
    LLE_C1_unroll_gamma=1e-9
    l_factor_1=Variable(torch.ones(1),requires_grad=True)
    #===============================================================================
    
    n_LLE=LLE_C1_unroll.shape[0]
    
    if LLE_yes_no==0:
        if low_rank_yes_no==0: # cholesky
            final_results[total_run_i_net,6]=nvariables
        else: # low rank
            final_results[total_run_i_net,6]=(n_train+n_train_net)*2
    else: # LLE
        if low_rank_yes_no==0: # cholesky
            final_results[total_run_i_net,6]=(nvariables+n_LLE+3)
        else: # low rank
            final_results[total_run_i_net,6]=((n_train+n_train_net)*2+n_LLE+3)

    print(f"Total number of parameters to be learnt: {final_results[total_run_i_net,6]}")
    
    if layer_i==1:
        y_md1,z_md1,lobpcg_fv_md1,M_d_md1,cL_md1,read_data_test1,alpha1=M_LOBPCG_LP_no_network(M_normalizer,optimizer_no_network,read_data_test,\
            torch.cat((b_ind,b_ind_train_net)),\
            db,\
            read_label_test,\
            n_feature,\
            M_d_0,\
            n_train+n_train_net,\
            metric_M_step,\
            n_train+n_train_net+n_test,\
            n_test,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_test,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_test,\
            numiter,\
            toler,Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            M_rec,\
            low_rank_yes_no,\
            LLE_yes_no)   
    elif layer_i==2:
        y_md1,z_md1,lobpcg_fv_md1,M_d_md1,cL_md1,read_data_test1,alpha1=M_LOBPCG_LP_no_network2(M_normalizer,optimizer_no_network,read_data_test,\
            torch.cat((b_ind,b_ind_train_net)),\
            db,\
            read_label_test,\
            n_feature,\
            M_d_0,\
            n_train+n_train_net,\
            metric_M_step,\
            n_train+n_train_net+n_test,\
            n_test,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_test,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_test,\
            numiter,\
            toler,Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            M_rec,\
            low_rank_yes_no,\
            LLE_yes_no)   
    else:
        y_md1,z_md1,lobpcg_fv_md1,M_d_md1,cL_md1,read_data_test1,alpha1=M_LOBPCG_LP_no_network3(M_normalizer,optimizer_no_network,read_data_test,\
            torch.cat((b_ind,b_ind_train_net)),\
            db,\
            read_label_test,\
            n_feature,\
            M_d_0,\
            n_train+n_train_net,\
            metric_M_step,\
            n_train+n_train_net+n_test,\
            n_test,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_test,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_test,\
            numiter,\
            toler,Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            M_rec,\
            low_rank_yes_no,\
            LLE_yes_no)               

    # print(M_d_0)    
    err_count_md,x_md=H_check_error(read_label_test,\
                    y_md1,\
                    z_md1,\
                    n_train+n_train_net+n_test,\
                    cL_md1,\
                    torch.cat((b_ind,b_ind_train_net)),\
                    numiter,\
                    toler,\
                    torch.cat((b_ind,b_ind_train_net)).shape[0]+torch.arange(0,n_test),\
                    n_train+n_train_net)
    # results[rsrng,2]=err_count_test
    results[rsrng+(K_i-1)*num_run0,2]=err_count_md     
    print(f"error_count model based test: {err_count_md}")       
    return results
  