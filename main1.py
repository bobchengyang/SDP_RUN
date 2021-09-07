import numpy as np
import numpy.matlib 
from sklearn.model_selection import KFold
import os

# torch.autograd.set_detect_anomaly(True)

import cvxpylayers.torch as ct
print(ct.__file__)

## import self-defined functions
from get_raw_data import get_raw_data
from ttvt_idx_no_val import ttvt_idx_no_val
from dataloader_normalized_standardization import dataloader_normalized_standardization
from GCN_experiments import GCN_experiments
from Unrolling_network_experiments import Unrolling_network_experiments
from model_based_experiments import model_based_experiments
from get_LLE_set_list import get_LLE_set_list
from metric_matrix_M_list import metric_matrix_M_list
from CNN_experiments import CNN_experiments
from MLP_experiments import MLP_experiments
from low_rank_approx_list import low_rank_approx_list

# translation from Matlab to Python of the GDPA implementation
alpha = 1e0 # self-loop weight adjustment
scalee=1/alpha # scaler for the cL
rho = 0 # GDPA left ends
sw = 0.5 # LP sw
metric_M_step=1e-2 # M diagonal step size
numiter=2e2 # lobpcg iter
toler=1e-4 # lobpcg tol
numiter_ep=20 # number of training epochs for the proposed network
numiter_nn=400 # number of training epochs for the black-box neural networks
lr=1e-2 # learning rate for all optimizers
std_or_01=1 # standardization: 1 || 0-1 minmax: 2 
feature_noise_level=1e-12 # feature noise level
num_run0=5 # shuffle the current fold of dataset twice
LLE_st=1e-3 # soft-thresholding parameter
random_number=0 # random seed for LLE initial weights
nneuron=32 # number of neurons in each layer of the competing neural nets
splitting_number1=300 # splitting number for k folds
splitting_number2=300 # splitting number for k folds
sparsity_threshold01=1.5 # sparsity threshold 0-1 for the initial lower triangular matrix
layer_i=1 # the number of unrolling layers in the unrolling network
gsm_i=0 # no graph smoothness term in the loss function: 0 || with this term in the loss function: 1
black_box_i=1 # choose the type of black boxes: GCN:1 || CNN:2 || MLP:3
M_normalizer_i=1 # choose the normalizer of the metric matrix M
low_rank_k=2 # choose the low rank prior
low_rank_yes_no=0 # low rank yes:1 || no:0
LLE_yes_no=0 # LLE yes:1 || no:0
train_all_n,train_net_n,val_n=0.6,0.2,0.1 # train_all,train_net,val

num_sum_dataset=np.array([690,683,768,862,1000,
                          206,270,583,345,556,
                          768,182,435,569,208,
                          2600,62]) # number of samples per dataset
dataset_i_vec=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # sequence number of each dataset
K_vec=np.concatenate([np.ceil(num_sum_dataset[0:15]/splitting_number1),np.ceil(num_sum_dataset[15:17]/splitting_number2)]) # number of folds of each dataset
final_results=np.zeros(shape=(17,9)) # get the final result in a 17x9 array (nerual network / unrolled network / model-based || 
                                     # number of trainable parameters / error rate (%) / error rate std)
further=0
total_run_i_net=0+further # current sequence number of a dataset (0-16)

for ee in dataset_i_vec: # loop for all 17 datasets
    dataset_i=ee+further # dataset number
    dataset_str,read_data_pd=get_raw_data(dataset_i) # read the dataset
    
    read_data_np=np.float32(np.array(read_data_pd)) # from pandas DataFrame to numpy Array
    read_data_np[read_data_np[:,-1]!=1,-1]=-1 # ensure binary labels 1's and -1's
    
    n_feature=read_data_np.shape[1]-1 # number of features of the dataset
    label=read_data_np[:,-1] # get the labels
    if int(K_vec[total_run_i_net])!=1:
        K=int(K_vec[total_run_i_net]) # number of folds of each dataset
        entire_flag=0
    else:
        K=2 # just for getting round with the train_idx test_idx loop
        entire_flag=1

    results=np.zeros(shape=(num_run0*K,3)) # get the subset of results into a num_run0*K by 3 array 
                                           # (number of trainable parameters / error rate (%) / error rate std)
    
    result_seq_i=0 # the current shuffle number of experiments
    
    kf=KFold(n_splits=K,shuffle=True,random_state=0) # k-fold cross-validation
    K_i=0 # the current fold number
    for train_idx, test_idx in kf.split(label): # we currently use the test_idx 
        K_i+=1 

        if entire_flag==0:
            read_data_np_i=read_data_np[test_idx,:] # get the current part of dataset to be used
        else:
            read_data_np_i=read_data_np
                
        for rsrng in range(num_run0): # loop for num_run0 times
            result_seq_i=result_seq_i+1
            print('==============================================================')
            print(f'======= dataset {dataset_i} fold {K_i} run number: {rsrng} =======')
            print('==============================================================')
            print('==============================================================')
            read_data_np_i_feature0=read_data_np[:,0:n_feature] # remove the nan's from the given dataset
            read_data_np_i_feature0[np.isnan(read_data_np_i_feature0)]=0 # remove the nan's from the given dataset
            np.random.seed(0) # set the random seed for the following line
            read_data_np_i_feature=np.float32(read_data_np_i_feature0+np.random.normal(0,feature_noise_level,read_data_np_i_feature0.shape)) # add some noise 
            # to the data to make sure the the following data standardization is valid and does not output nan's 
            read_data_np_i_label=read_data_np[:,-1] # get the label of the data
    
            n_sample=read_data_np_i.shape[0] # get the number of sample 
            # get indices for training_all, training_net, val, and test set
            n_train,n_train_net,n_val,n_test,b_ind,b_ind_train_net,b_ind_val,b_ind_test=\
                ttvt_idx_no_val(n_sample,train_all_n,train_net_n,val_n) # get number of samples for training/testing part of the data      
                
            read_data_train_net,read_data_val,read_data_test,\
            n_feature,\
            read_label_train_net,read_label_val,read_label_test,\
            Q_mask,nvariables,initial_Q_vec=\
                dataloader_normalized_standardization(read_data_np_i_feature,\
                                            read_data_np_i_label,\
                                            n_sample,\
                                            n_feature,\
                                            b_ind,\
                                            b_ind_train_net,\
                                            b_ind_val,\
                                            b_ind_test,\
                                            rsrng,
                                            std_or_01,\
                                            sparsity_threshold01) # feature normalization
            if M_normalizer_i==1:
                M_normalizer=n_feature
            else:
                M_normalizer=1    
            # ========method 1: pure blackbox method comparison================
            if black_box_i==1: # GCN
                results=GCN_experiments(random_number,\
                        initial_Q_vec,\
                        n_feature,\
                        nneuron,\
                        read_label_train_net,\
                        read_label_val,\
                        read_label_test,\
                        nvariables,\
                        total_run_i_net,\
                        numiter_nn,\
                        final_results,\
                        Q_mask,\
                        read_data_train_net,\
                        n_train,\
                        n_train_net,\
                        read_data_val,\
                        n_val,\
                        read_data_test,\
                        b_ind,\
                        n_test,\
                        results,\
                        K_i,\
                        rsrng,\
                        num_run0)                  
            elif black_box_i==2: # CNN
                results=CNN_experiments(random_number,\
                        initial_Q_vec,\
                        n_feature,\
                        nneuron,\
                        read_label_train_net,\
                        read_label_val,\
                        read_label_test,\
                        nvariables,\
                        total_run_i_net,\
                        numiter_nn,\
                        final_results,\
                        Q_mask,\
                        read_data_train_net,\
                        n_train,\
                        n_train_net,\
                        read_data_val,\
                        n_val,\
                        read_data_test,\
                        b_ind,\
                        n_test,\
                        results,\
                        K_i,\
                        rsrng,\
                        num_run0)              
            else: # MLP
                results=MLP_experiments(random_number,\
                        initial_Q_vec,\
                        n_feature,\
                        nneuron,\
                        read_label_train_net,\
                        read_label_val,\
                        read_label_test,\
                        nvariables,\
                        total_run_i_net,\
                        numiter_nn,\
                        final_results,\
                        Q_mask,\
                        read_data_train_net,\
                        n_train,\
                        n_train_net,\
                        read_data_val,\
                        n_val,\
                        read_data_test,\
                        b_ind,\
                        n_test,\
                        results,\
                        K_i,\
                        rsrng,\
                        num_run0)    
            #==================================================================
            
            #=====get the LLE set==============================================
            sim_mask,disim_mask,\
            LLE_mask_1,\
            LLE_C1_unroll,\
            LLE_C1_unroll_delta,\
            LLE_C1_unroll_gamma,\
            l_factor_1,\
            LLE_mask_2,\
            LLE_C2_unroll,\
            LLE_C2_unroll_delta,\
            LLE_C2_unroll_gamma,\
            l_factor_2,\
            LLE_mask_3,\
            LLE_C3_unroll,\
            LLE_C3_unroll_delta,\
            LLE_C3_unroll_gamma,\
            l_factor_3=get_LLE_set_list(n_train,n_train_net,read_label_train_net,b_ind,\
                         random_number,\
                         read_data_train_net,\
                         b_ind_train_net,\
                         n_feature,\
                         metric_M_step,\
                         LLE_st,\
                         lr)
            #==================================================================
            
            #=====get the metric matrix set====================================
            M_d_0,\
            M_d_0_,\
            M_d_0__,\
            M_rec1,\
            M_rec2,\
            M_rec3=metric_matrix_M_list(M_normalizer,random_number,\
                         initial_Q_vec,\
                         lr,\
                         read_data_train_net,\
                         b_ind,\
                         read_label_train_net,\
                         n_feature,\
                         n_train,\
                         metric_M_step,\
                         Q_mask,\
                         low_rank_k,\
                         low_rank_yes_no)  
            #====================================================================
            
            #====method 2: unrolling network=====================================
            results=Unrolling_network_experiments(M_normalizer,read_label_train_net,\
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
                                  LLE_yes_no)
            #====================================================================        
            
            #===method 3: model-based method (nips 2021 submission)==============
            results=model_based_experiments(M_normalizer_i,read_data_np_i_feature,\
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
                            LLE_yes_no)  
            #====================================================================
            
    results_percentage=results/n_test
    print(results_percentage)
    print(np.mean(results_percentage,axis=0))
    final_results[total_run_i_net,[1,4,7]]=np.mean(results_percentage,axis=0)
    final_results[total_run_i_net,[2,5,8]]=np.std(results_percentage,axis=0)
    total_run_i_net+=1
    print(final_results)
    if os.path.isdir('results')==False:
        os.mkdir("results")
    ee_further=ee+further
    f_textfile= open("results/GCN_MSE_one_layer_dataset_%s.txt" % ee_further,"w+")
    f_textfile.write(str(final_results))
    f_textfile.close()
    print("=========================================")  
print("=========================================")    
print("code execuated till this line") 
raise SystemExit(0)