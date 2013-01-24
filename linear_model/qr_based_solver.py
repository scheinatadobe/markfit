import numpy as np
import scipy.linalg as linalg


def solve(X,y,tolerance=1e-6) :
    n = X.shape[0]
    p = X.shape[1]

    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    R_diag = np.diag(R)
    firstSmall = np.where(R_diag < tolerance)[0]
    firstSmall = R_diag.shape[0] if firstSmall.shape[0] == 0 else firstSmall[0]
    print R.shape, np.diag(R)
    b     = np.dot(y,Q)[:firstSmall]
    #print "P is: ", P
    beta  = linalg.solve_triangular(R[:firstSmall,:firstSmall],b,lower=False)
    if firstSmall < R.shape[0] : beta = np.append(beta,np.zeros(R_diag.shape[0] - firstSmall))
    print "perm: ", permutation
    print "beta: ", beta
    beta[permutation]  = beta.copy() #non-atomic operation requires copy
    print "beta: ", beta
    R_small = R[:firstSmall,:firstSmall]
    XXn1  = linalg.pinv(np.dot(np.transpose(R_small),R_small))
    XXn1_diag = np.append(np.diag(XXn1),np.zeros(R.shape[1] - R_small.shape[1]))
    print "XXn1_diag: ", XXn1_diag
    #XXn1_diag[firstSmall:] = 0
    XXn1_diag[permutation] = XXn1_diag.copy()
    print y.shape, beta.shape, X.shape
    resid = y - np.dot(X,beta)
    SSE   = np.dot(resid,resid)
    return beta, XXn1_diag, SSE
