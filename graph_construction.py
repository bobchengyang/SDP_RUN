from get_graph_laplacian_variables_ready import get_graph_laplacian_variables_ready
from graph_laplacian import graph_laplacian

def graph_construction(feature,\
                       n_sample,\
                       n_feature,\
                       M):
    c=get_graph_laplacian_variables_ready(feature, n_sample, n_feature)
    L=graph_laplacian(n_sample,c,M)    
    return L