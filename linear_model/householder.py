""" 
This module handles various householder operations which we anticipate plugging into 
a stepwise regression functionality
"""
import unittest
import numpy as np
import math
from sys import stdout

def householder(x) :
    """ 
        x is a vector
        Compute a householder vector that is capable of zeroing out all but the 
        first cell of vector x
        return's v: the householder vector
        This implementation is based on Golub and Van Loan.
    """
    x = x.copy() / math.sqrt(np.dot(x,x))
    assert(x.dtype == np.float64)
    n = x.shape[0]
    sigma = np.dot(x[1:],x[1:])
    v = x.copy()
    v[0] = 1
    beta = 0
    if sigma != 0 :
        u = math.sqrt(x[0]*x[0] + sigma)
        if x[0] <= 0 :
            v[0] = x[0] - u
        else :
            v[0] = -sigma/(x[0] + u)
        beta = 2*v[0]*v[0] / (sigma + v[0]*v[0])
        v /= v[0]
    return beta, v
    
def apply_householder(X,beta,v) :
    """
    Employ vector v to make column i 0 BELOW the i'th row
    """
    if X.ndim == 1 :
        X = np.array([X]).T
    #print "shapes: ", X.shape, v.shape
    #print "apply house:", beta, X.shape, v.shape
    w = beta * np.dot(X.T,v)
    if X.shape[1] == 1 :
        w = np.array([w])
        rhs = (v * w).T
    else :
        rhs = np.outer(v,w)
    #print "returning shape: ", (X-rhs).shape, X.shape, rhs.shape
    stdout.flush()
    return X - rhs
    #print "w: ", w
    #print "dot: ", np.dot(v,X).shape, v.shape
    #print "comp", v.shape, (beta*np.dot(X.T,v)).T.shape
    #if X.shape == 1 :

    #print "returning shape: ",  (X - np.dot(v,beta*np.dot(X.T,v).T)).shape
    #return X - np.outer(v,beta*np.dot(v,X).T)

def QR_apply(X,y,active) :
    """ 
    X      - the design matrix
    y      - the response variable
    active - the set of variables that are active right now
    Returns A and QY, A[:,active] is R, however A[:,np.logical_not(active)] are the transformed columns of X
    which are not un upper triangular form.
    Qy are the transformed response variables.
    We do not return the householder vectors or Q.
    """
    assert y.ndim == 2, "In order to apply householder"
     
    A = X.copy()
    y = y.copy()
    ix = np.where(active)[0]
    for i,j in enumerate(ix) :
        #print "loop j:",j, "i: ",i
        beta, h  = householder(A[i:,j])
        A[i:,j:] = apply_householder(A[i:,j:],beta,h)
        y[i:]    = apply_householder(y[i:],beta,h)
    #print "A: "
    #print A
    stdout.flush()
    return A

def backSolve(R,y,active) :
    assert active.shape[0] == R.shape[1], "shapes should match."

class TestQRStepwiseSolver(unittest.TestCase) :
    def setUp(self) :
        pass
    
    def test_householder(self) :
        x = np.array([3,1,5,1],dtype=np.float64)
        beta,v = householder(x)
        #print "test_householder: ", beta, v
        answer = x - beta*np.dot(np.outer(v,v),x)  
        diffSum = np.sum(np.abs(answer - np.array([math.sqrt(np.dot(x,x)),0,0,0])))
        self.assertTrue(diffSum < 1e-8, diffSum)
    
    def test_QR(self) :
        matrix = np.array([[1,-1,4],[1,4,-2],[1,4,2],[1,-1,0]],dtype=np.float64)
        A=QR_apply(matrix,np.array([np.ones(4)]).T,active=np.ones(3,dtype=np.bool))
        for i in range(A.shape[1]) :
            self.assertTrue(abs(np.sum(A[i+1:,i])) < 1e-6)
    
    def test_QR2(self) :
        matrix = np.array([[12,-51,4],[6,167,-68],[-4,24,-41]])
        A = QR_apply(matrix,np.array([np.ones(3)]).T,active=np.ones(3,dtype=np.bool))
        #print A[1,1]
        self.assertTrue(abs(A[0,0]) - 14  < 1e-6)
        self.assertTrue(abs(A[0,1]) - 21  < 1e-6)
        self.assertTrue(abs(A[0,2]) - 14  < 1e-6)
        self.assertTrue(abs(A[1,1]) - 175 < 1e-6)
        self.assertTrue(abs(A[1,2]) - 70  < 1e-6)
        self.assertTrue(abs(A[2,2]) - 35  < 1e-6)
        self.assertTrue(abs(A[1,0]) + abs(A[2,0]) + abs(A[2,1]) < 1e-6)


       #add test
    def test_QRNoMiddle(self) :
        matrix = np.array([[1,-1,4],[1,4,-2],[1,4,2],[1,-1,0]],dtype=np.float64)
        active = np.ones(3,dtype=np.bool)
        active[1] = False
        A=QR_apply(matrix,np.array([np.ones(4)]).T,active)
        for i in range(A.shape[1]) :
            if not active[i] : continue
            self.assertTrue(abs(np.sum(A[i+1:,i])) < 1e-6)
        #add test


if __name__ == "__main__" :
   unittest.main() 
