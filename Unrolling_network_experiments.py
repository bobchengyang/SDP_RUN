import torch
import torch.optim as optim
import time 
from full_system import full_system
from H_check_error import H_check_error
from graph_construction import graph_construction
import torch.nn as nn
from get_LLE_set_list import get_LLE_set_list 
from full_system3 import full_system3
from full_system2 import full_system2
from LLE_relu_set import LLE_relu_set

def Unrolling_network_experiments(M_normalizer,read_label_train_net,\
                                  b_ind,\
                                  n_train,\
                                  n_train_net,\
                                  n_val,\
                                  n_test,\
                                  final_results,\
                                  total_run_i_net,\
                                  nvariables,\
                                  numiter_ep,\
                                  lr,\
                                  M_d_0,M_d_0_,M_d_0__,\
                                  LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,LLE_mask_1,l_factor_1,\
                                  LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,LLE_mask_2,l_factor_2,\
                                  LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,LLE_mask_3,l_factor_3,\
                                  read_data_train_net,\
                                  n_feature,\
                                  metric_M_step,\
                                  alpha,\
                                  rho,\
                                  sw,\
                                  scalee,\
                                  numiter,\
                                  toler,\
                                  Q_mask,\
                                  sim_mask,disim_mask,\
                                  b_ind_train_net,\
                                  random_number,\
                                  LLE_st,\
                                  read_data_test,\
                                  read_label_test,\
                                  results,\
                                  rsrng,\
                                  K_i,\
                                  num_run0,\
                                  layer_i,\
                                  gsm_i,\
                                  M_rec1,\
                                  M_rec2,\
                                  M_rec3,\
                                  low_rank_yes_no,\
                                  LLE_yes_no):
    db = 2*read_label_train_net[b_ind] # training_all labels x 2
    dz_plus_idx = db < 0
    dz_minus_idx = db > 0
    dz_ind_plus = b_ind[dz_plus_idx]
    dz_ind_minus = b_ind[dz_minus_idx] 
    
    y_known=torch.ones(n_train)
    y_unknown_train_net=torch.ones(n_train_net)
    y_unknown_val=torch.ones(n_val)
    y_unknown_test=torch.ones(n_test)
    y_np1=torch.Tensor([n_train])
    y_train_net=torch.cat((torch.cat((y_known,y_unknown_train_net),dim=0),y_np1),dim=0)
    y_val=torch.cat((torch.cat((y_known,y_unknown_val),dim=0),y_np1),dim=0)
    y_test=torch.cat((torch.cat((y_known,y_unknown_test),dim=0),y_np1),dim=0)
    z=-(db/2)  
    
    # initial evector 
    torch.random.manual_seed(0)
    lobpcg_fv_train_net=torch.randn(n_train+n_train_net+2,1)
    torch.random.manual_seed(0)
    lobpcg_fv_val=torch.randn(n_train+n_val+2,1)
    torch.random.manual_seed(0)
    lobpcg_fv_test=torch.randn(n_train+n_test+2,1)
    # M_d_0=torch.ones(1,n_feature).requires_grad_(True)
    loss_log=[]
    loss_label_log=[]
    loss_smoothness_log=[]
    error_count_train_net_log=[]
    error_count_val_log=[]
    # train a network
    n_LLE=LLE_C1_unroll.shape[0]
    if LLE_yes_no==0:
        if low_rank_yes_no==0: # cholesky
            if layer_i==1:
                final_results[total_run_i_net,3]=nvariables
            elif layer_i==2:
                final_results[total_run_i_net,3]=nvariables*2
            else:
                final_results[total_run_i_net,3]=nvariables*3
        else: # low rank
            if layer_i==1:
                final_results[total_run_i_net,3]=(n_train+n_train_net)*2
            elif layer_i==2:
                final_results[total_run_i_net,3]=(n_train+n_train_net)*2*2
            else:
                final_results[total_run_i_net,3]=(n_train+n_train_net)*2*3    
    else: # LLE
        if low_rank_yes_no==0: # cholesky
            if layer_i==1:
                final_results[total_run_i_net,3]=(nvariables+n_LLE+3)
            elif layer_i==2:
                final_results[total_run_i_net,3]=(nvariables+n_LLE+3)*2
            else:
                final_results[total_run_i_net,3]=(nvariables+n_LLE+3)*3
        else: # low rank
            if layer_i==1:
                final_results[total_run_i_net,3]=((n_train+n_train_net)*2+n_LLE+3)
            elif layer_i==2:
                final_results[total_run_i_net,3]=((n_train+n_train_net)*2+n_LLE+3)*2
            else:
                final_results[total_run_i_net,3]=((n_train+n_train_net)*2+n_LLE+3)*3    
 
    print(f"Total number of parameters to be learnt: {final_results[total_run_i_net,3]}")
    
    for _ in range(numiter_ep):
        # optimizer1=optim.Adam([M_d_0,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma],lr=lr)
        if LLE_yes_no==0:
            if low_rank_yes_no==0: # cholesky
                if layer_i==1:
                    optimizer_unrolling=optim.Adam([M_d_0],lr=lr)
                elif layer_i==2:
                    optimizer_unrolling=optim.Adam([M_d_0,M_d_0_],lr=lr)
                else:
                    optimizer_unrolling=optim.Adam([M_d_0,M_d_0_,M_d_0__],lr=lr)
            else: # low rank
                if layer_i==1:
                    optimizer_unrolling=optim.Adam([M_rec1],lr=lr)
                elif layer_i==2:
                    optimizer_unrolling=optim.Adam([M_rec1,M_rec2],lr=lr)
                else:
                    optimizer_unrolling=optim.Adam([M_rec1,M_rec2,M_rec3],lr=lr)        
        else: # LLE
            if low_rank_yes_no==0: # cholesky
                if layer_i==1:
                    optimizer_unrolling=optim.Adam([M_d_0,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1],lr=lr)
                elif layer_i==2:
                    optimizer_unrolling=optim.Adam([M_d_0,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                                                    M_d_0_,LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2],lr=lr)
                else:
                    optimizer_unrolling=optim.Adam([M_d_0,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                                                    M_d_0_,LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
                                                    M_d_0__,LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3],lr=lr)
            else: # low rank
                if layer_i==1:
                    optimizer_unrolling=optim.Adam([M_rec1,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1],lr=lr)
                elif layer_i==2:
                    optimizer_unrolling=optim.Adam([M_rec1,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                                                    M_rec2,LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2],lr=lr)
                else:
                    optimizer_unrolling=optim.Adam([M_rec1,LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                                                    M_rec2,LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
                                                    M_rec3,LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3],lr=lr)
     
        optimizer_unrolling.zero_grad() # zero the gradient buffers

        start = time.time()
        # print("runtime for LPs")
        print("runtime for SDPs")
        if layer_i==1:
            y_train_net1,z_train_net1,\
            y_train_net2,z_train_net2,\
            y_train_net3,z_train_net3,\
            lobpcg_fv_train_net1,\
            M_d_train_net1,\
            M_d_train_net1_,\
            M_d_train_net1__,\
            cL_train_net1,cL_train_net2,cL_train_net3,\
            read_data_train_net1,alpha_train_net1,LLE_C_matrix1,\
            LLE_C_matrix2,\
            LLE_C_matrix3=full_system(M_normalizer,read_data_train_net,\
            b_ind,\
            db,\
            read_label_train_net,\
            n_feature,\
            M_d_0,\
            M_d_0_,\
            M_d_0__,\
            n_train,\
            metric_M_step,\
            n_train_net,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_train_net,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_train_net,\
            numiter,\
            toler,\
            Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            LLE_C2_unroll,\
            LLE_mask_2,\
            l_factor_2,\
            LLE_C3_unroll,\
            LLE_mask_3,\
            l_factor_3,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
            LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
            M_rec1,\
            M_rec2,\
            M_rec3,\
            low_rank_yes_no,
            LLE_yes_no)
        elif layer_i==2:
            y_train_net1,z_train_net1,\
            y_train_net2,z_train_net2,\
            y_train_net3,z_train_net3,\
            lobpcg_fv_train_net1,\
            M_d_train_net1,\
            M_d_train_net1_,\
            M_d_train_net1__,\
            cL_train_net1,cL_train_net2,cL_train_net3,\
            read_data_train_net1,alpha_train_net1,LLE_C_matrix1,\
            LLE_C_matrix2,\
            LLE_C_matrix3=full_system2(M_normalizer,read_data_train_net,\
            b_ind,\
            db,\
            read_label_train_net,\
            n_feature,\
            M_d_0,\
            M_d_0_,\
            M_d_0__,\
            n_train,\
            metric_M_step,\
            n_train_net,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_train_net,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_train_net,\
            numiter,\
            toler,\
            Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            LLE_C2_unroll,\
            LLE_mask_2,\
            l_factor_2,\
            LLE_C3_unroll,\
            LLE_mask_3,\
            l_factor_3,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
            LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
            M_rec1,\
            M_rec2,\
            M_rec3,\
            low_rank_yes_no,
            LLE_yes_no)
        else:
            y_train_net1,z_train_net1,\
            y_train_net2,z_train_net2,\
            y_train_net3,z_train_net3,\
            lobpcg_fv_train_net1,\
            M_d_train_net1,\
            M_d_train_net1_,\
            M_d_train_net1__,\
            cL_train_net1,cL_train_net2,cL_train_net3,\
            read_data_train_net1,alpha_train_net1,LLE_C_matrix1,\
            LLE_C_matrix2,\
            LLE_C_matrix3=full_system3(M_normalizer,read_data_train_net,\
            b_ind,\
            db,\
            read_label_train_net,\
            n_feature,\
            M_d_0,\
            M_d_0_,\
            M_d_0__,\
            n_train,\
            metric_M_step,\
            n_train_net,\
            alpha,\
            rho,\
            sw,\
            scalee,\
            y_train_net,\
            z,\
            dz_ind_plus,\
            dz_ind_minus,\
            lobpcg_fv_train_net,\
            numiter,\
            toler,\
            Q_mask,\
            LLE_C1_unroll,\
            LLE_mask_1,\
            l_factor_1,\
            LLE_C2_unroll,\
            LLE_mask_2,\
            l_factor_2,\
            LLE_C3_unroll,\
            LLE_mask_3,\
            l_factor_3,\
            sim_mask,disim_mask,\
            LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
            LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
            LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
            M_rec1,\
            M_rec2,\
            M_rec3,\
            low_rank_yes_no,\
            LLE_yes_no)            
            
        end = time.time()
        # print(f"runtime for LPs: {end-start}s") 
        print(f"runtime for SDPs: {end-start}s") 

        err_count_train_net1,x_train_net1=H_check_error(read_label_train_net,\
                        y_train_net1,\
                        z_train_net1,\
                        n_train+n_train_net,\
                        cL_train_net1,\
                        b_ind,\
                        numiter,\
                        toler,\
                        b_ind.shape[0]+torch.arange(0,n_train_net),\
                        n_train)

        err_count_train_net2,x_train_net2=H_check_error(read_label_train_net,\
                        y_train_net2,\
                        z_train_net2,\
                        n_train+n_train_net,\
                        cL_train_net2,\
                        b_ind,\
                        numiter,\
                        toler,\
                        b_ind.shape[0]+torch.arange(0,n_train_net),\
                        n_train)

        err_count_train_net3,x_train_net3=H_check_error(read_label_train_net,\
                        y_train_net3,\
                        z_train_net3,\
                        n_train+n_train_net,\
                        cL_train_net3,\
                        b_ind,\
                        numiter,\
                        toler,\
                        b_ind.shape[0]+torch.arange(0,n_train_net),\
                        n_train)            
        print(f"error_count training 1: {err_count_train_net1}")
        print(f"error_count training 2: {err_count_train_net2}")
        print(f"error_count training 3: {err_count_train_net3}")
        print("=========================================")   
        
        read_data_train_net_train=read_data_train_net1[b_ind,:]
        L_loss=graph_construction(read_data_train_net_train, n_train+0, n_feature, M_d_train_net1)
        L_loss_=graph_construction(read_data_train_net_train, n_train+0, n_feature, M_d_train_net1_)
        L_loss__=graph_construction(read_data_train_net_train, n_train+0, n_feature, M_d_train_net1__)
        
        if layer_i==1:
            LLE_loss1=torch.linalg.norm(read_data_train_net-LLE_C_matrix1@read_data_train_net)**2
            loss_smoothness=read_label_train_net[b_ind].reshape(1,n_train)@L_loss@read_label_train_net[b_ind].reshape(n_train,1)
        elif layer_i==2:
            LLE_loss1=torch.linalg.norm(read_data_train_net-LLE_C_matrix1@read_data_train_net)**2\
                     +torch.linalg.norm(read_data_train_net-LLE_C_matrix2@read_data_train_net)**2
            loss_smoothness=read_label_train_net[b_ind].reshape(1,n_train)@L_loss@read_label_train_net[b_ind].reshape(n_train,1)\
                           +read_label_train_net[b_ind].reshape(1,n_train)@L_loss_@read_label_train_net[b_ind].reshape(n_train,1)           
        else:
            LLE_loss1=torch.linalg.norm(read_data_train_net-LLE_C_matrix1@read_data_train_net)**2\
                     +torch.linalg.norm(read_data_train_net-LLE_C_matrix2@read_data_train_net)**2\
                     +torch.linalg.norm(read_data_train_net-LLE_C_matrix3@read_data_train_net)**2
            loss_smoothness=read_label_train_net[b_ind].reshape(1,n_train)@L_loss@read_label_train_net[b_ind].reshape(n_train,1)\
                           +read_label_train_net[b_ind].reshape(1,n_train)@L_loss_@read_label_train_net[b_ind].reshape(n_train,1)\
                           +read_label_train_net[b_ind].reshape(1,n_train)@L_loss__@read_label_train_net[b_ind].reshape(n_train,1)
            
        loss_label=nn.MSELoss()(x_train_net1[b_ind_train_net], read_label_train_net[b_ind_train_net])*1e0\
                  +nn.MSELoss()(x_train_net2[b_ind_train_net], read_label_train_net[b_ind_train_net])*1e0\
                  +nn.MSELoss()(x_train_net3[b_ind_train_net], read_label_train_net[b_ind_train_net])*1e0
 
        if gsm_i==1:
            loss = loss_label+loss_smoothness
        else:
            loss = loss_label
        # print(f"mse_gst_weight: {mse_gst_weight}")
        # loss = nn.MSELoss()(x_val*1e5, read_label*torch.abs(x_val)*1e5)
        loss_label_log.append(loss_label.item())
        loss_smoothness_log.append(loss_smoothness.item())
        loss_log.append(loss_label.item()+loss_smoothness.item())
        error_count_train_net_log.append(err_count_train_net1.detach())
        # error_count_val_log.append(err_count_val.detach())
        
        print(f"loss label: {loss_label.item()}")
        print(f"loss smoothess: {loss_smoothness.item()}")
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer_unrolling.step() # Does the update  
        
        LLE_C_relu=nn.ReLU()
        LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
        LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
        LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3=LLE_relu_set(LLE_C_relu,\
                 LLE_C1_unroll,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,l_factor_1,\
                 LLE_C2_unroll,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,l_factor_2,\
                 LLE_C3_unroll,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,l_factor_3)
        
        # print(f"M1: {M_d_0}")
        # print(f"M2: {M_d_0_}")
        # print(f"M3: {M_d_0__}")
        # print(f"delta1: {LLE_C1_unroll_delta} gamma1: {LLE_C1_unroll_gamma}")
        # print(f"delta2: {LLE_C2_unroll_delta} gamma2: {LLE_C2_unroll_gamma}")
        # print(f"delta3: {LLE_C3_unroll_delta} gamma3: {LLE_C3_unroll_gamma}")
        asdf=1
        # optimizer2.step() # Does the update  
        # print(list(net1.named_parameters()))
        
    #=====get the LLE set==============================================
    sim_mask,disim_mask,\
    LLE_mask_1,\
    LLE_C1_unroll,\
    LLE_C1_unroll_delta_test,\
    LLE_C1_unroll_gamma_test,\
    l_factor_1_test,\
    LLE_mask_2,\
    LLE_C2_unroll,\
    LLE_C2_unroll_delta_test,\
    LLE_C2_unroll_gamma_test,\
    l_factor_2_test,\
    LLE_mask_3,\
    LLE_C3_unroll,\
    LLE_C3_unroll_delta_test,\
    LLE_C3_unroll_gamma_test,\
    l_factor_3_test=get_LLE_set_list(n_train,n_test,read_label_test,b_ind,\
                 random_number,\
                 read_data_test,\
                 b_ind.shape[0]+torch.arange(0,n_test),\
                 n_feature,\
                 metric_M_step,\
                 LLE_st,\
                 lr)
    #==================================================================
    
    # test the above network
    if layer_i==1:
        y_test1,z_test1,\
        y_test2,z_test2,\
        y_test3,z_test3,\
        lobpcg_fv_test1,M_d_test1,M_d_test1_,M_d_test1__,\
            cL_test1,cL_test2,cL_test3,\
            read_data_test1,alpha_test1,\
            LLE_C_matrix1_test,\
            LLE_C_matrix2_test,\
            LLE_C_matrix3_test=\
            full_system(M_normalizer,read_data_test,\
        b_ind,\
        db,\
        read_label_test,\
        n_feature,\
        M_d_0,\
        M_d_0_,\
        M_d_0__,\
        n_train,\
        metric_M_step,\
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
        LLE_C2_unroll,\
        LLE_mask_2,\
        l_factor_2,\
        LLE_C3_unroll,\
        LLE_mask_3,\
        l_factor_3,\
        sim_mask,disim_mask,\
        LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
        LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
        LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
        M_rec1,\
        M_rec2,\
        M_rec3,\
        low_rank_yes_no,\
        LLE_yes_no)
    elif layer_i==2:
        y_test1,z_test1,\
        y_test2,z_test2,\
        y_test3,z_test3,\
        lobpcg_fv_test1,M_d_test1,M_d_test1_,M_d_test1__,\
            cL_test1,cL_test2,cL_test3,\
            read_data_test1,alpha_test1,\
            LLE_C_matrix1_test,\
            LLE_C_matrix2_test,\
            LLE_C_matrix3_test=\
            full_system2(M_normalizer,read_data_test,\
        b_ind,\
        db,\
        read_label_test,\
        n_feature,\
        M_d_0,\
        M_d_0_,\
        M_d_0__,\
        n_train,\
        metric_M_step,\
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
        LLE_C2_unroll,\
        LLE_mask_2,\
        l_factor_2,\
        LLE_C3_unroll,\
        LLE_mask_3,\
        l_factor_3,\
        sim_mask,disim_mask,\
        LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
        LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
        LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
        M_rec1,\
        M_rec2,\
        M_rec3,\
        low_rank_yes_no,\
        LLE_yes_no)
    else:
        y_test1,z_test1,\
        y_test2,z_test2,\
        y_test3,z_test3,\
        lobpcg_fv_test1,M_d_test1,M_d_test1_,M_d_test1__,\
            cL_test1,cL_test2,cL_test3,\
            read_data_test1,alpha_test1,\
            LLE_C_matrix1_test,\
            LLE_C_matrix2_test,\
            LLE_C_matrix3_test=\
            full_system3(M_normalizer,read_data_test,\
        b_ind,\
        db,\
        read_label_test,\
        n_feature,\
        M_d_0,\
        M_d_0_,\
        M_d_0__,\
        n_train,\
        metric_M_step,\
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
        LLE_C2_unroll,\
        LLE_mask_2,\
        l_factor_2,\
        LLE_C3_unroll,\
        LLE_mask_3,\
        l_factor_3,\
        sim_mask,disim_mask,\
        LLE_C1_unroll_delta,LLE_C1_unroll_gamma,\
        LLE_C2_unroll_delta,LLE_C2_unroll_gamma,\
        LLE_C3_unroll_delta,LLE_C3_unroll_gamma,\
        M_rec1,\
        M_rec2,\
        M_rec3,\
        low_rank_yes_no,\
        LLE_yes_no)        

    err_count_test,x_test=H_check_error(read_label_test,\
                    y_test3,\
                    z_test3,\
                    n_train+n_test,\
                    cL_test3,\
                    b_ind,\
                    numiter,\
                    toler,\
                    b_ind.shape[0]+torch.arange(0,n_test),\
                    n_train)
    results[rsrng+(K_i-1)*num_run0,1]=err_count_test
    print(f"error_count test: {err_count_test}")
    print("=========================================")          
    return results