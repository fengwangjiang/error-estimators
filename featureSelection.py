"""
    
Select features from data for classification and error estimation.

"""

import numpy as np
def t_test(X, y, d):
    """
    Feature selection using t-test
    ------------------------------
    X -> dataset
    y -> label
    d -> selected feature size
    output -> feature index
    """
    ind1 = y==0
    ind2 = y==1
    X1 = X[ind1]
    X2 = X[ind2]
    m1 = X1.mean(axis=0)
    s1 = X1.std(axis=0)
    m2 = X2.mean(axis=0)
    s2 = X2.std(axis=0)
    t = abs(np.divide((m1-m2),(s1+s2)))#t score  
    ind = np.argsort(t)
    FeaInd = ind[-d:]
    return FeaInd
if __name__ == '__main__':
    print __doc__
