#module to fit least squares problem
#we are starting to implement some primitives here that will eventually grow into a doolittle/
#sweep approach

import numpy as np
from solver_common import pp
from sys import stdout

def adjust(A,p,k,zero_list,tolerance=1e-8) :
    B = A[k,k]
    #assert abs(B) > tolerance
    #print "B:", B, tolerance
    if abs(B) <= tolerance :
        #print "SINGULAR"
        stdout.flush()
        zero_list.append(k)
        #A[k] = 0
        #A[:,k] =0
        return

    A[k] /= B
    for i in range(A.shape[0]) :
        if i == k : continue
        B = A[i,k]
        A[i] -= B*A[k]
    

def findBestPivotCandidate(A,minK) :
    assert minK < A.shape[0]-1
    bestIA  = minK
    bestVal = A[bestIA,bestIA]
    
    for k in range(minK,A.shape[0]-1) :
        if abs(A[k,k]) > bestVal :

            bestIA  = k
            bestVal = abs(A[bestIA,bestIA])
    return bestIA

def swap(A,E,i,j) :
    temp = E[i]
    E[i] = j
    E[j] = temp
    temp = A[i].copy()
    A[i] = A[j]
    A[j] = temp

def unswapify(Z,E) :
    """
    Z - a matrix
    E - a desciption of a set of swap operations that must be undone
    """
    for i,other in reverse([ (i,val) for val in enumerate(E.flat) ]) :
        temp     = Z[i]
        Z[i]     = Z[other]
        Z[other] = temp

def solve(A) :
    """
    A is an (2*p+1 x 2*p+1) augmented matrix consisting of
    X'X|X'y|I  
    y'X|y'y|0
    which get's solved into
    I|b  |(X'X)^-1
    0|ESS|-b'
    The transformation happens in-place
    """
    p = A.shape[0] - 1 #number of predictors
    assert A.shape[1] == 2*p + 1, "Dimension of axis 1 should be 2*p+1"
    residuals = np.zeros(p) 
    E = np.arange(p,dtype=np.int64)
    zero_list = []
    for i in range(p) :
        #print "A: "
        #pp(A)
        #print "A diag: ", np.diag(A)
        k = findBestPivotCandidate(A,i)
        swap(A,E,i,k)
        #print "swapping ", i , "with ", k
        #print "A after: "
        #pp(A)
        #print "A after: ", np.diag(A)
        adjust(A,p,i,zero_list)
        #pp(A)
    for k in zero_list :
        A[k] = 0
        A[:,k] = 0
    


if __name__ == "__main__" :
    import json_loader
    output_directory = "/d0/data/auto-mpg-test"
    local=True
    XX,keysXX    = json_loader.loadXXFromFile(output_directory + "/standardize/xx/part-00000",
                                                local=True)
    Xy,keysXy,yy = json_loader.loadXyFromFile(output_directory + "/standardize/xy/part-00000",
                                                local=True)
    p = XX.shape[0]
    print XX.shape, Xy.shape
    A = np.append(XX,np.array([Xy]).T,1)
    A = np.append(A,np.identity(XX.shape[0]),1)
    bottom = np.append(Xy,np.array([yy]),0)
    bottom = np.append(bottom,np.zeros(p),0)
    print "A.shape: ", A.shape, "bottom.shape: ", bottom.shape
    A = np.append(A,[bottom],0)
    solve(A)
    I = np.dot(XX,A[:p,p+1:])
    I[np.abs(I) < 1e-10] = 0
    print I[:10,:10]

