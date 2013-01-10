""" 
As we incrementally build out the solver from ADJUST->SWEEP->SWEEP + speudoinverse
here are some common routines.
""" 

def findBestPivotCandidate(A,minK) :
    assert minK < A.shape[0]-1
    bestIA  = minK
    bestVal = A[bestIA,bestIA]

    for k in range(minK,A.shape[0]-1) :
        if abs(A[k,k]) > bestVal :
            bestIA  = k
            bestVal = A[bestIA,bestIA]
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

def pp(A,precision=3) :
    fmt = "%d.%df"%(precision*2 + 2,precision)
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            print ("%" + fmt)%A[i,j],

            if j < A.shape[1] : print " ",
        print

