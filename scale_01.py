def scale_01(feature,n_feature):
    for i in range(n_feature):
        feature_i=feature[:,i]
        feature[:,i]=0+(feature_i-min(feature_i))/(max(feature_i)-min(feature_i))
    return feature