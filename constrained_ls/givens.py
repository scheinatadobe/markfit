#Implementation of Givens Rotations
#see Golub and Van Loan, 3rd edition

from math import sqrt
import unittest

def computeGivensParameters(a,b) :
    """ 
    [c   s]' [a]  = [r]
    [-s  c]  [b]  = [0]
    given a,b this function returns c,s
    """
    if b == 0.0 :
        return 1.0,0.0
    if abs(b) > abs(a) :
        tau = -float(a) /b
        s   = 1/sqrt(1+tau*tau)
        c   = s*tau
        return c,s
    tau = -float(b)/a
    c   =  1/sqrt(1+tau*tau)
    s   =  c*tau
    return c,s

def preApplyGivensRotation(c,s,A,i,k) :
    """
    Perfroms the givens rotation operation in-place on A([i,k],:).
    """
    for j in range(A.shape[1]) :
        tau1 = A[i,j]
        tau2 = A[k,j]
        A[i,j] = c*tau1 - s*tau2
        A[k,j] = s*tau1 + c*tau2

#The post procedure is commented out because I have not yet
#written any tests for it.
#def postApplyGivensRotation(c,s,A,i,k) :
#    for j in range(A.shape[0]) :
#        tau1 = A[j,i]
#        tau2 = A[j,k]
#        A[j,i] = c*tau1 - s*tau2
#        A[j,k] = s*tau1 + c*tau2

class TestComputeGivensParameters(unittest.TestCase) :
    #I think a proper test is to use the Givens procedure to do a basic QR decomposition
    def setUp(self) :
        pass
    def checkResult(self,X) :
        #makes sure the matrix is upper triangular with non-zeros on diag
        for i in range(X.shape[0]) :
            self.assertTrue(np.abs(X[i,i]) > 1e-5)
            self.assertTrue( np.all(np.abs(X[i+1:,i]) < 1e-6))

    def test2(self) :
        #A test where the 3'rd row has an unwanted value in the second column
        X = np.array([[1,2,3],[0,4,5],[0,-9,11]])
        c,s = computeGivensParameters(X[1,1],X[2,1])
        preApplyGivensRotation(c,s,X,1,2)
        self.checkResult(X)

    def test1(self) :
        #A test where the 2nd row has an unwanted value in the first column
        X   = np.array([[1,2,3],[4,-5,6],[0,0,3]])
        c,s = computeGivensParameters(X[0,0],X[1,0])
        preApplyGivensRotation(c,s,X,0,1)
        self.checkResult(X)
    
    def test3(self) :
        #A test where the 2nd and third row has unwanted values in the first column
        #in addition to an unwanted value in the 3rd row, second column
        X   = np.array([[1,2,3],[4,-5,6],[-1,0,3]])
        c,s = computeGivensParameters(X[1,0],X[2,0])
        preApplyGivensRotation(c,s,X,1,2)
        c,s = computeGivensParameters(X[0,0],X[1,0])
        preApplyGivensRotation(c,s,X,0,1)
        c,s = computeGivensParameters(X[1,1],X[2,1])
        preApplyGivensRotation(c,s,X,1,2)
        self.checkResult(X)

    
if __name__ == '__main__':
    import numpy as np
    unittest.main()

