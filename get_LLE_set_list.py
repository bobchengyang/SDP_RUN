from LLE_delta_gamma_mask import LLE_delta_gamma_mask
from get_LLE_set import get_LLE_set

def get_LLE_set_list(n_train,n_train_net,read_label_train_net,b_ind,\
                     random_number,\
                     read_data_train_net,\
                     b_ind_train_net,\
                     n_feature,\
                     metric_M_step,\
                     LLE_st,\
                     lr):
    
    sim_mask,disim_mask=LLE_delta_gamma_mask(n_train,n_train_net,read_label_train_net,b_ind) # get the similar and dis-similar locations of the similarity matrix 

    LLE_mask_1,\
    LLE_C1_unroll,\
    LLE_C1_unroll_delta,\
    LLE_C1_unroll_gamma,\
    l_factor_1=get_LLE_set(random_number,\
        n_train,\
        n_train_net,\
        read_data_train_net,\
        b_ind,\
        b_ind_train_net,\
        read_label_train_net,\
        n_feature,\
        metric_M_step,\
        LLE_st,\
        lr)
    LLE_mask_2,\
    LLE_C2_unroll,\
    LLE_C2_unroll_delta,\
    LLE_C2_unroll_gamma,\
    l_factor_2=get_LLE_set(random_number,\
        n_train,\
        n_train_net,\
        read_data_train_net,\
        b_ind,\
        b_ind_train_net,\
        read_label_train_net,\
        n_feature,\
        metric_M_step,\
        LLE_st,\
        lr)
    LLE_mask_3,\
    LLE_C3_unroll,\
    LLE_C3_unroll_delta,\
    LLE_C3_unroll_gamma,\
    l_factor_3=get_LLE_set(random_number,\
        n_train,\
        n_train_net,\
        read_data_train_net,\
        b_ind,\
        b_ind_train_net,\
        read_label_train_net,\
        n_feature,\
        metric_M_step,\
        LLE_st,\
        lr)    
    return sim_mask,disim_mask,\
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
           l_factor_3
        