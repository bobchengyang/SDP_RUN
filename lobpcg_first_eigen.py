import torch

def lobpcg_first_eigen(initial_H,lobpcg_fv,numiter,toler):
    evalue,evector=torch.lobpcg(-initial_H, k=1, B=None, X=lobpcg_fv, n=1, iK=None, \
             niter=numiter, tol=toler, largest=None, method="ortho", \
             tracker=None, ortho_iparams=None, ortho_fparams=None, \
             ortho_bparams=None)    
    return evalue,evector