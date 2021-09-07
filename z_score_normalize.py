import numpy as np

def z_score_normalize(feature_in,b_ind):
    mean_TRAIN_set_0=np.mean(feature_in,axis=0)
    std_TRAIN_set_0=np.std(feature_in,axis=0)
    feature_in_0=(feature_in-mean_TRAIN_set_0)/std_TRAIN_set_0
    feature_train_12=np.sqrt(np.sum(feature_in_0**2,axis=1))
    feature_out=feature_in_0/feature_train_12.reshape(b_ind.size(0),1)
    return feature_out