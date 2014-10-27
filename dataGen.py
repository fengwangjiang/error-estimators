"""
    
Generate random datasets according to different models from "high dimensional bolstered error estimation" 2011 by Chao Sima, et al.

"""
import numpy as np
def data_gen1(n, D, d, dt, std):
    """

    Model 1. 

    n -> sample size
    D -> feature size
    d -> marker size
    dt -> markers means with +/- dt.
    std -> standard deviation of markers
    output -> two nxD datasets
    """
    n1, n2 = n, n
    m1 = np.zeros(shape=[n1, D])
    m2 = np.zeros(shape=[n2, D])
    std1 = np.ones(shape=[n1, D])
    std2 = np.ones(shape=[n2, D])
    m1[:, 0:d] = -dt# markers with means +-dt
    m2[:, 0:d] = dt
    std1[:, 0:d] = std
    std2[:, 0:d] = std 
    r1 = std1 * np.random.standard_normal(size=[n1, D])
    r2 = std2 * np.random.standard_normal(size=[n1, D])
    X1 = m1 + r1
    X2 = m2 + r2
    return (X1, X2)

def data_gen2(n, D, d, dt, std, c):
    """
    Model 2.

    n -> sample size
    D -> feature size
    d -> marker size
    dt -> markers means with +/- dt.
    std -> standard deviation of markers
    c -> a constant, I and cI are covariance matrices
    output -> two nxD datasets
    """
    n1, n2 = n, n
    m1 = np.zeros(shape=[n1, D])
    m2 = np.zeros(shape=[n2, D])
    std1 = np.ones(shape=[n1, D])
    std2 = np.ones(shape=[n2, D])
    m1[:, 0:d] = -dt# markers with means +-dt
    m2[:, 0:d] = dt
    std1[:, 0:d] = std
    std2[:, 0:d] = std*c 
    r1 = std1 * np.random.standard_normal(size=[n1, D])
    r2 = std2 * np.random.standard_normal(size=[n1, D])
    X1 = m1 + r1
    X2 = m2 + r2
    return (X1, X2)
if __name__ == '__main__':
    print __doc__
