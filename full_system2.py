from M_LOBPCG_LP import M_LOBPCG_LP

def full_system2(M_normalizer,read_data_train_net,\
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
                lobpcg_fv,\
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
                LLE_yes_no):
    y_1,z_1,lobpcg_fv_1,M_d_1,cL1,read_data1,alpha1,LLE_C_matrix1=M_LOBPCG_LP(M_normalizer,read_data_train_net,\
    b_ind,\
    db,\
    read_label_train_net,\
    n_feature,\
    M_d_0,\
    n_train,\
    metric_M_step,\
    n_train+n_train_net,\
    n_train_net,\
    alpha,\
    rho,\
    sw,\
    scalee,\
    y_train_net,\
    z,\
    dz_ind_plus,\
    dz_ind_minus,\
    lobpcg_fv,\
    numiter,\
    toler,0,Q_mask,LLE_C1_unroll,LLE_mask_1,l_factor_1,sim_mask,disim_mask,LLE_C1_unroll_delta,LLE_C1_unroll_gamma,M_rec1,low_rank_yes_no,\
    LLE_yes_no)
        
    y_2,z_2,lobpcg_fv_2,M_d_1_,cL2,read_data2,alpha2,LLE_C_matrix2=M_LOBPCG_LP(M_normalizer,read_data1,\
    b_ind,\
    db,\
    read_label_train_net,\
    n_feature,\
    M_d_0_,\
    n_train,\
    metric_M_step,\
    n_train+n_train_net,\
    n_train_net,\
    alpha1,\
    rho,\
    sw,\
    scalee,\
    y_1,\
    z_1,\
    dz_ind_plus,\
    dz_ind_minus,\
    lobpcg_fv_1,\
    numiter,\
    toler,1,Q_mask,LLE_C2_unroll,LLE_mask_2,l_factor_2,sim_mask,disim_mask,LLE_C2_unroll_delta,LLE_C2_unroll_gamma,M_rec2,low_rank_yes_no,\
    LLE_yes_no)
        
    # y_3,z_3,lobpcg_fv_3,M_d_1__,cL3,read_data3,alpha3,LLE_C_matrix3=M_LOBPCG_LP(M_normalizer,read_data2,\
    # b_ind,\
    # db,\
    # read_label_train_net,\
    # n_feature,\
    # M_d_0__,\
    # n_train,\
    # metric_M_step,\
    # n_train+n_train_net,\
    # n_train_net,\
    # alpha2,\
    # rho,\
    # sw,\
    # scalee,\
    # y_2,\
    # z_2,\
    # dz_ind_plus,\
    # dz_ind_minus,\
    # lobpcg_fv_2,\
    # numiter,\
    # toler,1,Q_mask,LLE_C3_unroll,LLE_mask_3,l_factor_3,sim_mask,disim_mask,LLE_C3_unroll_delta,LLE_C3_unroll_gamma,M_rec3,low_rank_yes_no,\
    # LLE_yes_no)
        
    # y_4,z_4,lobpcg_fv_4,M_d_1___,cL,read_data4,alpha4=M_LOBPCG_LP(read_data3,\
    # b_ind,\
    # db,\
    # read_label_train_net,\
    # n_feature,\
    # M_d_0___,\
    # n_train,\
    # metric_M_step,\
    # n_train+n_train_net,\
    # n_train_net,\
    # alpha3,\
    # rho,\
    # sw,\
    # scalee,\
    # y_3,\
    # z_3,\
    # dz_ind_plus,\
    # dz_ind_minus,\
    # lobpcg_fv_3,\
    # numiter,\
    # toler,1)
        
    # y_5,z_5,lobpcg_fv_5,M_d_1____,cL,read_data5,alpha5=M_LOBPCG_LP(read_data4,\
    # b_ind,\
    # db,\
    # read_label_train_net,\
    # n_feature,\
    # M_d_0____,\
    # n_train,\
    # metric_M_step,\
    # n_train+n_train_net,\
    # n_train_net,\
    # alpha4,\
    # rho,\
    # sw,\
    # scalee,\
    # y_4,\
    # z_4,\
    # dz_ind_plus,\
    # dz_ind_minus,\
    # lobpcg_fv_4,\
    # numiter,\
    # toler,1)
    return y_1,z_1,\
           y_2,z_2,\
           y_2,z_2,\
           lobpcg_fv_2,M_d_1,M_d_1_,M_d_1_,cL1,cL2,cL2,read_data2,alpha2,LLE_C_matrix1,LLE_C_matrix2,LLE_C_matrix2