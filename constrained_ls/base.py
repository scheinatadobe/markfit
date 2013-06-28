import numpy as np

def __LQ(A) :
    """ 
    LQ code for representing linear constraints 
    If A is not full rank, the underlying QR decomposition code in numpy will 
    throw an exception
    """
    #Why do we do this? Q represents the nullspace of A
    Q,L = numpy.linalg.qr(A.T)
    Q = np.transpose(Q)
    L = np.transpose(L)
    return L, Q

def __add_constraint_LQ(L,Q,constraint) :
    """ 
    On further reflection of the underlying algorithm, let's implement the sorounding
    context and then fill this in
    """
    pass

def __del_constraint_LQ(L,Q,constraint) :
    """
    Fill in once the surounding context is set up
    """
    pass

def __is_feasible_point(A,b,C,d,w,verbose=True) :
    """
    Verify that w satisfies
    Aw >= b
    Cw  = d.
    """

    if A is not None and np.any(np.dot(A,w) < b) :
        return False
    if C is not None and not np.allclose(np.dot(C,w),d) :
        return False
    return True
    
def __has_converged(X,y,A,b,C,d,w,active,g,H,Z,lamb) :
    #active - the active constraints
    #g      - the gradient vector
    #H      - the Hessian
    #Z      - the nullspace of the constraints
    #lamb   - the lambda multiplier vector

    #I1: Ax >= b with \hat{A} x = \hat{b}
    #we count on active set def to enforce the latter
    #why am I checking constraints in a convergence test? Lack of feasibility
    #should result in a real termination?
    if not np.all(np.dot(A,w) >= b) :
        return False
    if not np.allclose(np.abs(np.dot(C,w), d)) :
        return False
    if not np.allclose(np.abs(np.dot(A[:,active],w[active]) , b[active])) :
        return False
    #I2: Z'g(x) = 0
    if not np.allclose(np.dot(Z.T,g),0) :
        return False
    #I3: \lambda_i >= 0
    if not np.all(lamb >= 0) :
        return False
    #

def __ineq_ls(X,y,A,b,w0,aIx,verbose=True) :
    """
    Minimize || Xw - y||^2 
    s.t.  Aw >= b
    and   A[aIx]w = b
    The nature of the second constraint is that equality constraints have been folded into
    the A matrix, but the indices of those constraints must always be in the working set.
    """
    g  = 0      #starting gradient vector
    Ix = aIx[:] #the working set
    while not has_converged() :
        #LI2: decide whether to continue minimizimng in the current subspace or whether
        #to delete a constraint from the working set.
        if continue_minimizing :
            #compute feasible direction
            #comptue step length
            if step_length_acceptable :
                update_estimate
                continue
            else :
                #add appropriate constraint to the working set
                #choose alpha to meet constraint
                update_estimate
                continue
        else : # delete a constraint from the working set
            #choose a constraint to be deleted s
            #delete s from I_k
            continue
    return w


def ls(X,y,A,b,C,d,w0,verbose=True) :
    """
    Minimize || Xw - y||^2 
    s.t.  Aw >= b
    and   Cw  = d.
    
    w0 is a given feasible solution, which is used as a starting point.
    If A,b,C,d may be passed as None when there are no applicable constraints.
    """
    assert (A is not None and b is not None) or (A is None and b is None)
    assert (C is not None and d is not None) or (C is None and d is None)
    
    #The approach we use requries  a feasible starting point, and guarantees that all
    #subsequent updates are also feasible
    w_0 = __is_feasible_point(A,b,C,d,w0,verbose=verbose)
    Z,_ = __LQ(A) #Z represents the null space of A
    if A is None and C is None :
        #this is an ordinary least squares problem
        return np.linalg.lstsq(X,y)
    elif A is None :
        #use Golub and Van Loan Chapter 12 method
        Q,R = np.linalg.qr(C) #numpy's qr will throw an error if there is insufficient rank
        p = C.shape[1]
        assert R.shape[1] == C.shape[1]
        w = scipy.linalg.solve_triangular(R[:p,:p,d)
        X = np.dot(X,Q)
        z = np.linalg.lstsq(X[:,p+1:], np.dot(y - X[:,:p],w))
        return np.dot(Q[:,:p],w) + np.dot(Q[:,p+1:],z)
    else :
       aIx = np.zeros(A.shape[0],dtype==bool)
       if B is not None :  #add equality constraints to the working set
           aIx.append(np.ones(B.shape[0],dtype=bool))
       #with equality constraints encoded as working set, we can merge A and B matrices
       #and pass along to a specialized routine.
       return __ineq_ls(X,y,np.vstack((A,B)),np.vstack(b,d),w0,aIx,verbose=True)

if __name__ == "__main__" : 
    A  = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
    b  = np.array([10,20,30])
    C  = None
    d  = None 
    is_feasible_point(A,b,C,d)
