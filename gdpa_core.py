import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

def gdpa_core(n_sample,\
              n_train,\
              n_test,\
              scaled_M,\
              scaled_factors,\
              rho,\
              db,\
              cL,\
              b_ind,\
              alpha,\
              sw,\
              dz_ind_plus,\
              dz_ind_minus):
    
    scaled_M_n_sample=scaled_M[0:n_sample,0:n_sample]
    scaled_M_n=scaled_M_n_sample-torch.diag(torch.diag(scaled_M_n_sample))
   
    # Define and solve the CVXPY LP problem.
    x_lp=cp.Variable((1,n_sample+1+2*n_train)) # n+1+2M
    c_lp=cp.Parameter((1,n_sample+1+2*n_train))
    c_ynp1=torch.ones(n_sample+1)
    c=torch.cat((torch.cat((c_ynp1,db),dim=0),torch.zeros(n_train)),dim=0)
    c_tch=torch.reshape(c,(1,n_sample+1+2*n_train)).requires_grad_(True)
    
    # get constraint
    d_n_sample=(torch.diag(cL)-torch.sum(torch.abs(scaled_M_n),dim=1)).detach()
    
    constraints = []
    mask_x_lp_np1=np.zeros((1,n_sample+1+2*n_train)) # does not need to be updated within the following loop!
    # row 1 to row M
    for n_train_i in range(0,n_train): 
        mask_x_lp=np.zeros((1,n_sample+1+2*n_train)) # needs to be updated within the loop!
        mask_x_lp_abs1=np.zeros((1,n_sample+1+2*n_train)) # needs to be updated within the loop! 
        mask_x_lp_abs2=np.zeros((1,n_sample+1+2*n_train)) # needs to be updated within the loop!
        mask_x_lp[0,n_train_i]=-1
        mask_x_lp_abs1[0,n_sample+1+n_train_i]=1
        mask_x_lp_abs1[0,n_sample+1+n_train+n_train_i]=-1
        mask_x_lp_abs2[0,n_sample+1+n_train_i]=-1
        mask_x_lp_abs2[0,n_sample+1+n_train+n_train_i]=-1
        constraints.append(x_lp[:,n_train_i]>=torch.trace(cL).detach())
        # b_i>0, i.e., z_i<0
        if db[n_train_i].sign()==1: 
            # constraints.append(x_lp[:,n_sample+1+n_train_i]<=0)
            # constraints.append(x_lp[:,n_sample+1+n_train+n_train_i]>=0)
            scalars=torch.abs(scaled_factors[n_train_i,n_sample])
            mask_x_lp_np1[0,n_sample+1+n_train+n_train_i]=torch.abs(scaled_factors[n_sample,n_train_i])
        # b_i<0, i.e., z_i>0
        else: 
            # constraints.append(x_lp[:,n_sample+1+n_train_i]>=0)
            # constraints.append(x_lp[:,n_sample+1+n_train+n_train_i]>=0)
            scalars=torch.abs(scaled_factors[n_train_i,n_sample+1])
            mask_x_lp_np1[0,n_sample+1+n_train+n_train_i]=torch.abs(scaled_factors[n_sample+1,n_train_i])
        mask_x_lp[0,n_sample+1+n_train+n_train_i]=scalars
        constraints.append(cp.sum(cp.multiply(mask_x_lp,x_lp))<=d_n_sample[n_train_i]-rho)
        constraints.append(cp.sum(cp.multiply(mask_x_lp_abs1,x_lp))<=0)
        constraints.append(cp.sum(cp.multiply(mask_x_lp_abs2,x_lp))<=0)
    mask_x_lp_np1[0,n_sample]=-1   
    constraints.append(cp.sum(cp.multiply(mask_x_lp_np1,x_lp))<=-rho)  # row N+1   
        
    for n_i in range(n_train,n_sample): # row M+1 to row N
        mask_x_lp=np.zeros(shape=(1,n_sample+1+2*n_train))
        mask_x_lp[0,n_i]=-1
        constraints.append(cp.sum(cp.multiply(mask_x_lp,x_lp))<=d_n_sample[n_i]-rho) 
        constraints.append(x_lp[:,n_i]>=torch.trace(cL).detach())
    
    # formulate the problem
    prob=cp.Problem(cp.Minimize(cp.sum(cp.multiply(c_lp,x_lp))),constraints)
    assert prob.is_dpp()
    cvxpylayer=CvxpyLayer(prob,parameters=[c_lp],variables=[x_lp])
    
    # solve the problem
    # prob.solve(verbose=True, solver=cp.SCS)
    # c_tch=torch.randn(1,n_sample+1+2*n_train,requires_grad=True)
    cvxpylayer_scale=1
    while "solution" not in locals():
        try:
            solution, = cvxpylayer(c_tch,solver_args={"verbose":False,\
                                                      "scale":cvxpylayer_scale,\
                                                      "eps":1e-6,\
                                                      "max_iters":1000})
        except:
            cvxpylayer_scale=cvxpylayer_scale*1e1
    # solution, = cvxpylayer(c_tch,solver_args={"eps":1e-5,\
    #                                           "max_iters":10000,\
    #                                           "verbose":True,\
    #                                           "acceleration_lookback":10,\
    #                                           "rho_x":1e0,\
    #                                           "alpha":1e0,\
    #                                           "scale":1e0,\
    #                                           "normalize":bool(1)})
    # get 7 after three LP    
    # solution, = cvxpylayer(c_tch,solver_args={"eps":1e-5,\
    #                                           "max_iters":10000,\
    #                                           "verbose":True,\
    #                                           "acceleration_lookback":10,\
    #                                           "rho_x":1e-3,\
    #                                           "alpha":1e-0,\
    #                                           "scale":1e-0,\
    #                                           "normalize":bool(1)})        
    # solution, = cvxpylayer(c_tch,solver_args={"eps":1e-3,\
    #                                           "max_iters":1000,\
    #                                           "verbose":True,\
    #                                           "acceleration_lookback":10,\
    #                                           "rho_x":1e-4,\
    #                                           "alpha":1e-2,\
    #                                           "scale":1e-4,\
    #                                           "normalize":bool(1)})
    
    # compute the gradient of the sum of the solution with respect to c
    # solution.sum().backward()   
    
    #x_lp_solution=solution.detach().numpy()
    y0=solution[0,0:n_sample+1]   
    z0=solution[0,n_sample+1:n_sample+1+n_train]
    #current_obj=np.sum(np.multiply(c,x_lp_solution))
    current_obj=torch.sum(torch.multiply(c,solution))
    return y0,z0,current_obj