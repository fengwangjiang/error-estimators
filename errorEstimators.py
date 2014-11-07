"""
Error estimators: resubstitution, cross validation with 10 folds, bolstered resubstitution, bootstrap 0.632.

Classifiers: linear svm, kernel svm, lda, knn, cart.
"""
import numpy as np
from sklearn import svm, cross_validation, lda 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from featureSelection import t_test
from kernelSize import mean_min_dists_1sigma, mean_min_dists_Dsigma

def linear_svm_clf():
    return svm.SVC(kernel='linear')
def kernel_svm_clf():
    return svm.SVC()
def lda_clf():
    return lda.LDA()
def knn_clf(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=n_neighbors)
def cart_clf(max_depth=3):
    return DecisionTreeClassifier(max_depth=max_depth)
classifiers = {"lsvm" : linear_svm_clf,
        "svm" : kernel_svm_clf,
        "lda" : lda_clf,
        "knn" : knn_clf,
        "cart" : cart_clf,
        }
def crossValidation(X_train, y_train, d, nFolds, clf_opt="linear"):
    """
    svm cross validation (need feature selection for each XD_train)
    ---------------------------------------------------------------
    X_train -> nxD
    y_train -> n
    d -> selected feature size
    nFolds -> by default, 10 folds.
    clf_opt -> classifier options
    output -> error rate
    """
    kf = cross_validation.KFold(len(y_train),n_folds=nFolds,indices=True)
    errs = []
    for train_index, test_index in kf:
        Xcv10_train = X_train[train_index]
        Xcv10_test = X_train[test_index]
        ycv10_train = y_train[train_index]
        ycv10_test = y_train[test_index]
        FeaInd_cv10 = t_test(Xcv10_train,ycv10_train,d)    
        Xdcv10_train = Xcv10_train[:,FeaInd_cv10]
        Xdcv10_test = Xcv10_test[:,FeaInd_cv10]
        # clf = svm.SVC(kernel='linear')
        clf = classifiers[clf_opt]()
        clf.fit(Xdcv10_train,ycv10_train)
        err = 1-clf.score(Xdcv10_test, ycv10_test)    
        errs.append(err)
    err_cv10 = np.array(errs).mean()
    return err_cv10

def bootstrap(X_train, y_train, d, nIter=100, random_state=0, clf_opt="linear"):
    """
    svm bootstrap 0 and 632.
    ------------------------
    X_train -> nxD
    y_train -> n
    d -> selected feature size
    clf_opt -> classifier options
    output -> err_bs0, err_bs632
    """
    bs = cross_validation.Bootstrap(len(y_train),n_iter=nIter,random_state=random_state)
    errs0 = []
    errs632 = []
    for train_index, test_index in bs:
        Xbs_train = X_train[train_index]
        Xbs_test = X_train[test_index]
        ybs_train = y_train[train_index]
        ybs_test = y_train[test_index]
        FeaInd_bs = t_test(Xbs_train,ybs_train,d)    
        Xdbs_train = Xbs_train[:,FeaInd_bs]
        Xdbs_test = Xbs_test[:,FeaInd_bs]
        # clf = svm.SVC(kernel='linear')
        clf = classifiers[clf_opt]()
        clf.fit(Xdbs_train,ybs_train)
        err0 = 1-clf.score(Xdbs_test, ybs_test)
        err_bs_resub = 1-clf.score(Xdbs_train, ybs_train)
        err632 = (1-0.632)*err_bs_resub+0.632*err0
        errs0.append(err0)
        errs632.append(err632)
    err_bs0 = np.array(errs0).mean()
    err_bs632 = np.array(errs632).mean()
    return (err_bs0, err_bs632)

def brNew(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10, nNeighbor=3, clf_opt="linear"):
    """
    svm bolstered resubstitution, D-sigma
    -------------------------------------
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    nNeighbor -> # of neighbors to calculate sigma
    clf_opt -> classifier options
    output -> error rate
    """
    # clf = svm.SVC(kernel='linear')
    clf = classifiers[clf_opt]()
    clf.fit(Xd_train,yd_train)    
    neighbors=nNeighbor
    sigs = mean_min_dists_Dsigma(X_train,y_train,neighbors)
    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_brNEWD = np.array(errs).mean()
    return err_brNEWD

def brOLD(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10, clf_opt="linear"):
    """
    svm bolstered resubstitution, original 1-sigma
    ----------------------------------------------
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    clf_opt -> classifier options
    output -> error rate
    """
    nNeighbor = 1
    # clf = svm.SVC(kernel='linear')
    clf = classifiers[clf_opt]()
    clf.fit(Xd_train,yd_train)
    sigs = mean_min_dists_1sigma(X_train,y_train, nNeighbor)
    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_brOLD = np.array(errs).mean()
    return err_brOLD

def bolsteredResub(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10, nNeighbor=3, newORold="new", clf_opt="linear"):
    """
    svm bolstered resubstitution, original 1-sigma, and new D-sigma.
    ----------------------------------------------
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    nNeighbor -> # of neighbors to calculate sigma
    newORold -> select new or old bolstered resubstitution
    clf_opt -> classifier options
    output -> error rate
    """
    # clf = svm.SVC(kernel='linear')
    clf = classifiers[clf_opt]()
    clf.fit(Xd_train,yd_train)
    if newORold == "new":
        neighbors = nNeighbor
        sigs = mean_min_dists_Dsigma(X_train,y_train,neighbors)
    else:
        neighbors = 1
        sigs = mean_min_dists_1sigma(X_train,y_train, neighbors)
    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_br = np.array(errs).mean()
    return err_br
if __name__ == '__main__':
    print __doc__

