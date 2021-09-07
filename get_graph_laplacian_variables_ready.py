def get_graph_laplacian_variables_ready(feature,n_sample,n_feature):
    a=feature.reshape((n_sample,1,n_feature))
    c=(a-a.permute(1,0,2)).reshape((n_sample**2,n_feature))
    return c