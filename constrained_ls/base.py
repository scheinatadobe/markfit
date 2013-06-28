import numpy as np
import ../linear_model.householder as hh

class IndexMap :
    """ 
    Keep track of which row correponds to which constraint in some original 
    contraint matrix A.  A is imagined to be the complete set of constraints in a problem
    consiting of both equality and inequality constraints.  We use this data structure to keep
    track of which rows of A are encoded in some other matrix, and also the oder of those rows.
    """
    def __init__(self,lst) :
        """ lst should be a list of integers """
        assert(type(lst) == type([]))
        self.forward = lst.copy()
        self.back = dict( [val,i : for i,val in enumerate( lst ) ] )

    def append(self, index) :
        """
        add a new row to the constraint matrix 
        """
        assert index not in self.back
        self.forward.append(index)
        self.back[index] = len(self.forward) - 1
    
    def delete(self, index) :
        """
        index refers to the original constraint matrix, not the current active set.
        delete a row from the constraint matrix
        """
        i = self.back[index]
        del self.forward[index]
        del delf.back[index]


def LQ(A) :
    """
    An function which returns the LQ decomposition 
    If A is not full rank, the underlying QR decomposition code in numpy will throw an exception.
    The caller may want to keep track on the row indices going forward to support addition and deletion of rows.
    """
    Q,R = numpy.linalg.qr(A.T) #Q now represents the null space of A
    Q   = np.transpose(Q)
    L   = np.transpose(R)
    return L, Q, IndexMap( [i for i in range(A.shape[0])] )

def __add_constraint_LQ(L,Q,indexMap,new_row_to_append,new_row_index) :
    # Practical Optimization, pg 172.
    # My householder code is written for QR rather than LQ, so for now we convert back to this form
    # to perform intermediate computations
    t   = L.shape[0]
    Q   = np.transpose(Q)
    R   = np.transpose(L)
    R   = np.hstack(R,np.dot(new_row_to_append,Q))
    beta,v = hh.householder(R[t+1,t+1:])
    R[t+1,t+1:] = hh.apply_householder(R[t+1,t+1:],beta,v)
    Q[t+1,t+1:] = hh.apply_householder(Q[t+1,t+1:],beta,v)
    Q   = np.transpose(Q)
    L   = np.transpose(R)
    indexMap.append(new_row_to_index)
    return L,Q, indexMap

def __del_constraint_LQ(L, Q, row_indices, row_index_to_delete) :
    # Practical Optimization, pg 172-173.
    R   = np.transpose(L)
    Q   = np.transpose(Q)
    #in transpose view, we are deleting a column
    #R = np.hstack(R[:,row_index_to_delete],R[:,row_index_to_delete+1:])
    row_indices = np.delete(row_indices, row_index_to_delete) 
    #now all rows at row_index_to_delete and higher feature an unwanted super-diagonal element
    for i in range(row_index_to_delete,R.shape[1]) :
        c,s = computeGivensParameters(R[i-1,i],R[i,i]
        #TODO: check arguments of preeApplyGivensRotation
        preApplyGivensRotation(c,s,A,i,i+1) 
    return np.transpose(R), np.transpose(Q), np.delete(row_indices,row_index_to_delete)

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

def __has_converged(X,y,A,b,w,active,g,Z,lamb) :
    #active - the active constraints
    #g      - the gradient vector
    #H      - the Hessian
    #Z      - the nullspace of the constraints
    #lamb   - the lambda multiplier vector

    #I1: Ax >= b with \hat{A} x = \hat{b}
    #we count on active set def to enforce the latter
    #why am I checking constraints in a convergence test? Lack of feasibility
    #should result in a real termination?
    #AIS: double check sufficient conditions
    if not np.all(np.dot(A,w) >= b) :
        return False
    if not np.allclose(np.abs(np.dot(A[:,active],w[active]) , b[active])) :
        return False
    #I2: Z'g(x) = 0
    if not np.allclose(np.dot(Z.T,g),0) :
        return False
    #I3: \lambda_i >= 0
    if not np.all(lamb >= 0) :
        return False
    return True 

def find_feasible_direction(X,y,w,Z) :
    """ TODO: rewrite using QR decomposition to avoid re-computation. """
    #TODO: possitble typo in Gill book in R implementation means we have the wrong sign
    #Computes  Newton direction that can be taken either step 1 or else shortened 
    #to the neareast constraint.
    return np.dot(Z,np.linalg.lstsq(np.dot(X,Z),np.dot(X,w) - y))

def compute_step_length(X,y,w,Z,A,b,active,p_z) :
    #TODO: can I unit test this
    #TODO: double check that 1.0 is the optimimum in this direction (chk normalization of direction)
    length = np.norm(p_z)
    assert(length >= 1e-8)
    direction = p_z / length
    inactive = np.logical_not(active)
    constrainable = np.logical_and(inactive, np.dot(A[inactive],direction) < 0 )
    ix = np.where(constrainable)
    if not np.any(inactive) : 
        return length
    gammas = list((b[constrainable] - np.dot(A[constrainable],w))/ np.dot(A[constrainable], \
                        direction))
    gammaIx = np.argmin(gammas)
    if gammas[gammaIx] < 1.0) :
        return (gammaIx,gammas[gammaIx],)
    return (None, 1.0,)

def compute_lagrange_multipliers(X,y,A,b,w,verbose=True) :
    """ 
    There is a clever way to solve this using Householder computations,
    using the QT we have already computed, but for now we'll implement the brute force approach.
    """
    g = np.dot(np.dot(X.T,X),w) - np.dot(y,w)
    return np.linalg.lstsq((np.transpose(A.t), g)
     

def __ineq_ls(X,y,A,b,alwaysActive,verbose=True) :
    """
    Minimize || Xw - y||^2 
    s.t.  Aw >= b
    and   A[alwaysActive]w = b
    The nature of the second constraint is that equality constraints have been folded into
    the A matrix, but the indices of those constraints must always be in the working set.
    aIx are those constraints that must always be active; these are equality constraints from the
    original problem.
    """
    #The idea behind the implementation is to ensure the candidate solution alwasy satisfies
    #the active constraints.  When we add a constraint it is because a step length would cause
    #us to break the constraint... so we add the constraint and shorten the constraint accordingly.
    #when we remove the constraint, this is true by default.  Below, we compute a solution w which
    #obeys all constraints as a starting point.  Don't worry, we try to remove a constraint in the
    #first iteration of the loop.
    i = 0
    active  = np.ones(A.shape[0],dytype=np.bool)
    L,Q     = LQ(A) # start with all constraints active.
    Z       = Q[:,A.shape[0]]
    w       = solve_equality_ls(X,y,A,b) 
    #p_k     = find_feasible_direction(X,y,w,Z)
    #alpha   = compute_step_length(X,y,w,Z,A,b,active,p_k) 
    #AIS: do I want to keep sending the complete matrix Q around or shrink and grow it?
    while True : #not has_converged(X,y,A,b,w,active,g,Z) :
        #loop invariant:  we have already solved optimimum for the current working set. 
        i += 1 #just for debugging
        # the question is, is there a constraint we can remove, or should we quit
        lambdas = compute_lagrange_multipliers(X,y,A,b,w)
        j = np.argmin(lambdas)
        if lambdas[j] < 0 :
            #delete constraint j
            
        else :
            return w

        
        p_k     = find_feasible_direction(X,y,w,Z)
        alpha   = compute_step_length(X,y,w,Z,A,b,active,p_k)      
        solve_equality_ls(X,y,A,b) 
        #LI2: decide whether to continue minimizimng in the current subspace or whether
        #to delete a constraint from the working set. I do not believe this test is necessary
        #in our special case, because we always step as far as we can in the current subspace.
        #if continue_minimizing :
            #p_z = find_feasible_direction(X,y,w,Z)
            #compiute step length
            #if step_length_acceptable :
            #    update_estimate
            #    continue
            #else :
        #add appropriate constraint to the working set
        #choose alpha to meet constraint
        
        p_k     = find_feasible_direction(X,y,w,Z)
        alpha   = compute_step_length(X,y,w,Z,A,b,active,p_k)     
        continue
        else : # delete a constraint from the working set
            #choose a constraint to be deleted s
            #delete s from I_k
            continue
    return w

def solve_equality_ls(X,y,C,d) :
    #use Golub and Van Loan Chapter 12 method to solve a least squares problem which has only
    #quality constraints
    Q,R = np.linalg.qr(C) #numpy's qr will throw an error if there is insufficient rank
    p = C.shape[1]
    assert R.shape[1] == C.shape[1]
    w = scipy.linalg.solve_triangular(R[:p,:p,d)
    X = np.dot(X,Q)
    z = np.linalg.lstsq(X[:,p+1:], np.dot(y - X[:,:p],w))
    return np.dot(Q[:,:p],w) + np.dot(Q[:,p+1:],z)

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
        return solve_equality_ls(X,y,C,d)     
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
