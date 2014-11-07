#!/usr/bin/python
"""
Compare D-sigma, 2-sigma, 1-sigma bolstered resubstitution error estimators.
----------------------------------------------------------------------------
Feature selection: t-test
Classification rule: linear svm, kernel svm, knn, cart, lda.
Error estimators: resubstitution, cross validation with 10 folds, bolstered resubstitution, bootstrap 0.632.
"""

from datetime import datetime
import numpy as np
import sys
import dataGen, featureSelection, kernelSize, errorEstimators
from sklearn import cross_validation
def syntheticSim():
    tstart = datetime.now()
    n = 300
    Ds = [100] #, 200, 400]
    d0s = [15] #, 30]
    dt = 0.3
    stds= [1.0]
    ds = [5] #, 10, 15, 20]
    clf_opts = ["lsvm", "svm", "lda", "knn", "cart"]
    N = 20 
    neighbors = [1]#, 3, 5]
    lneighbors = len(neighbors)
    if lneighbors == 1:
        header = "true\tresub\tcv10\tloo\tbs0\tbs632\tbrNEWD_k1\tbrOLD\n"
    elif lneighbors == 3:
        header = "true\tresub\tcv10\tloo\tbs0\tbs632\tbrNEWD_k1\tbrNEWD_k3\tbrNEWD_k5\tbrOLD\n"
    for D in Ds:
        for d0 in d0s:
            for std in stds:
                X1, X2 = dataGen.data_gen1(n,D,d0,dt,std)
                y1 = np.zeros((X1.shape[0]))
                y2 = np.ones((X2.shape[0]))
                XD = np.vstack((X1,X2))
                y = np.concatenate((y1,y2))
                for d in ds:
                    FeaInd_d = featureSelection.t_test(XD,y,d)
                    for clf_opt in clf_opts:
                        fn = "%s_dt%.1f_std_%.1f_D%d_d0%d_d%d.txt"%(clf_opt, dt, std, D, d0, d)
                        f = open(fn,'w')
                        f.write(header)
                        for i in range(N):
                            # randomly split data into train and test
                            X_train, X_test, y_train, y_test = \
                                    cross_validation.train_test_split\
                                    (XD, y,test_size=0.80)
                            Xd_train, Xd_test, yd_train, yd_test = \
                                    X_train[:,FeaInd_d], X_test[:,FeaInd_d],\
                                    y_train, y_test        
                            # clf = svm.SVC(kernel='linear')
                            clf = errorEstimators.classifiers[clf_opt]()
                            clf.fit(Xd_train,yd_train)
                            
                            # svm true error
                            err_true = 1-clf.score(Xd_test, yd_test)
                            # svm resubsitution
                            err_resub = 1-clf.score(Xd_train,yd_train)
                            # svm cv10
                            err_cv10 = errorEstimators.crossValidation(X_train, y_train, d, nFolds=10, clf_opt=clf_opt)
                            # svm loo
                            err_loo = errorEstimators.crossValidation(X_train, y_train, d, nFolds=len(y_train), clf_opt=clf_opt)
                            # svm bootstrap 0 and 632
                            err_bs0, err_bs632 = errorEstimators.bootstrap(X_train, y_train, d, nIter=100, random_state=0, clf_opt=clf_opt)
                            # svm bolstered resubstitution NEW 
                            # neighbors = [1]#, 3, 5]
                            err_brNEWDs = np.zeros(len(neighbors))
                            for iter in range(len(neighbors)):
                                neighbor = neighbors[iter]
                                err_brNEWDs[iter] = errorEstimators.bolsteredResub(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10, nNeighbor=neighbor, newORold="new", clf_opt=clf_opt)
                            err_brOLD = errorEstimators.bolsteredResub(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10, nNeighbor=neighbor, newORold="old", clf_opt=clf_opt)
                            if lneighbors == 1:
                                format1 = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" \
                                        % (err_true,err_resub,err_cv10,err_loo, err_bs0,err_bs632,err_brNEWDs[0],err_brOLD)
                                format = format1
                            elif lneighbors == 3:
                                format2 = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" \
                                        % (err_true,err_resub,err_cv10,err_loo, err_bs0, \
                                        err_bs632,err_brNEWDs[0],err_brNEWDs[1],err_brNEWDs[2],err_brOLD)
                                format = format2
                            f.write(format)
                        f.close()
                        msg = fn + " is being written..."
                        print msg
    tend = datetime.now()
    print tend - tstart
if __name__ == '__main__':
    print __doc__
    # syntheticSim()
