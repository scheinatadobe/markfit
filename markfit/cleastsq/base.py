if __name__ == "__main__" and __package__ is None:
        __package__ = "pyleastsq.cleastsq"
import numpy as np
#from pyleastsq import householder as hh
from ..leastsq import householder as hh
import givens
import unittest 
import scipy.linalg
from sys import stderr, stdout

# Need 2 more non-trivial examples to test
# add and delete

class IndexMap :
    """ 
    Keep track of which row correponds to which constraint in some original 
    contraint matrix A.  A is imagined to be the complete set of constraints in a problem
    consiting of both equality and inequality constraints. What we are keeping track of
    is which constraints are part of the active set.  The active set constraints may be
    represented in some matrix Z, and we require a mapping to keep track of which 
    constraints are represented and which rows of Z correspond.
    """
    def __init__(self,lst) :
        """ 
            lst should be a list of integers 
            self.forward: maps Z matrix rows to original constraint matrix rows
            self.back:    maps constraint matrix rows to Z matrix rows
        """
        assert(type(lst) == type([]))
        self.forward = lst[:] 
        self.back = dict( [(val,i)  for i,val in enumerate( lst ) ] )

    def __len__(self) :
        return len(self.forward)

    def append(self, index) :
        """
        add a new row to the constraint matrix
        index represents the index with respect to an original constraint matrix
        """
        assert index not in self.back, self.back
        self.forward.append(index)
        self.back[index] = len(self.forward) - 1
    
    def delete(self, index) :
        """
        index refers to the original constraint matrix, not the current active set.
        delete a row from the constraint matrix
        """
        i = self.back[index]
        del self.forward[i]
        foundKey = None
        for j,k in enumerate(self.back) :
            v = self.back[k]
            if v > i : 
                self.back[k] = v-1
            elif v == i :
                foundKey = k
        assert foundKey is not None
        del self.back[foundKey]

class TestIndexMap(unittest.TestCase) :
    def test_create1342(self) :
        lst = [1,3,4,2]
        ixMap = IndexMap(lst)
        self.assertTrue(ixMap.forward == lst)
        self.assertTrue(ixMap.back[1] == 0)
        self.assertTrue(ixMap.back[2] == 3)
        self.assertTrue(ixMap.back[3] == 1)
        self.assertTrue(ixMap.back[4] == 2,ixMap.back)
    
    def test_append(self) :
        lst = [1,3,4,2]
        ixMap = IndexMap(lst)
        ixMap.append(5)
        self.assertTrue(ixMap.forward == lst + [5])
        self.assertTrue(ixMap.back[5] == 4)
    
    def test_delete(self) :
        lst = [1,3,4,2]
        ixMap = IndexMap(lst)
        ixMap.delete(4)
        self.assertTrue([1,3,2] == ixMap.forward)
        self.assertTrue(ixMap.back[1] == 0)
        self.assertTrue(ixMap.back[3] == 1,ixMap.back[3])
        self.assertTrue(ixMap.back[2] == 2,ixMap.back[2])
        self.assertTrue(len(ixMap.back) == 3)
    
def LQ(A) :
    """
    An function which returns a modified LQ decomposition.
    The function actually returns (L 0) rather than just L.
    This is suboptimal, but should be easier to get correct in a first implementation.
    If A is not full rank, the underlying QR decomposition code in numpy will throw an exception.
    A index Mapper is returned so that the caller can keep track of which rows of L 
    correspond to which rows of A going forward.
    """

    if A.shape[0] == 0 :
        return np.zeros([0,A.shape[1]]),np.diag([1 for i in range(A.shape[1])]), IndexMap([])
    
    Q,R = scipy.linalg.qr(A.T) #Q now represents the null space of A
    Q   = np.transpose(Q)
    L   = np.transpose(R)
    return L, Q, IndexMap( [i for i in range(A.shape[0])] )

class TestLQ(unittest.TestCase) :
    def test_Empty(self) :
        A = np.zeros([0,3])
        L,Q,ixMap = LQ(A)
        self.assertTrue(L.shape == (0,A.shape[1],))
        self.assertTrue(Q.shape == (A.shape[1],A.shape[1],))
          

def _add_constraint_LQ(L,Q,indexMap,new_row_to_append,constraint_row_index) :
    """
    Form a new L, Q matrix by adding a new row to the bottom.
    constraint_row_index describes the row in the original constraint matrix (0-indexed)
    """
    # Practical Optimization, pg 172.
    # My householder code is written for QR rather than LQ, so for now we convert back to this form
    # to perform intermediate computations
    
    t   = L.shape[0]
    Q   = np.transpose(Q)
    R   = np.zeros([L.shape[1],L.shape[0]+1])
    R[:,:-1] = np.transpose(L)
    newcol = np.dot(new_row_to_append,Q)
    R[:,t] = newcol
    beta,v = hh.householder(R[t,t:])
    R[t,t:] = hh.apply_householder(R[t,t:],beta,v)
      
    Q[t,t:] = hh.apply_householder(Q[t,t:],beta,v)
    Q   = np.transpose(Q)
    L   = np.transpose(R)
    indexMap.append(constraint_row_index)
    return L,Q

def _del_constraint_LQ(L, Q, indexMap, constraint_index_to_delete) :
    # Practical Optimization, pg 172-173.
    #constraint_index_to_delete describes the row of the original constraint matrix to delete
    row_index_to_delete = indexMap.back[constraint_index_to_delete]
    R   = np.transpose(np.delete(L,row_index_to_delete,0))
    Q   = np.transpose(Q)
    stdout.flush()
    indexMap.delete(constraint_index_to_delete)
    #in transpose view, we have deleted a column.  Now we have to straighten out the
    #columns which are no-longer upper triangular.  The problematic rows are at
    #row_index_to_delete and higher, and these rows feature a single unwanted super-diagonal
    #element.
    for i in range(row_index_to_delete, R.shape[1]) :
        c,s = givens.computeGivensParameters(R[i-1,i],R[i,i])
        #TODO: check arguments of preeApplyGivensRotation
        givens.preApplyGivensRotation(c,s,R,i,i+1) 
    return np.transpose(R), np.transpose(Q)

class TestConstraintManipulation(unittest.TestCase) :
    """ 
    Tests the code related to adding and deleting constraints
    """
    def setUp(self) :
        #The idea behind A is that we have a full identity matrix
        self.A = np.array([[1,0,0],[0,1,0],[0,0,1]])
        #B is an incomplete identity matrix which we could make complete by adding 1 more constraint
        self.B = np.array([[1,0,0],[0,0,1]])
        
        #C motivation: let's create a two parameter regression model and then 
        # constrain the slope to be positive
        self.C = np.array([[1,0],[0,1]])
        #self.C     = np.array([[0,1]])

    def testLQ(self) :
        #Q,R = scipy.linalg.qr(self.B,mode='full')
        #Q,R = scipy.linalg.qr(self.B.T,mode='full')
        
        L,Q,imap = LQ(self.B)
        self.assertTrue(Q.shape[0] == Q.shape[1])
        self.assertTrue(L.shape[0] == self.B.shape[0])
        self.assertTrue(L.shape[1] == self.B.shape[1])

    def testDelA(self) :
        #start with full constraint matrix
        L,Q,indexMap = LQ(self.A)
        #It turns out that L and Q are both the identify for this case, making it 
        #perhaps not the most interesting test case.
        #delete 2nd constraint
        L,Q = _del_constraint_LQ(L,Q,indexMap,2)
        newA = np.dot(L,Q)
        expected_result = np.array([[1,0,0],[0,1,0]])
        self.assertTrue(np.allclose(expected_result, newA))
    
    def testAddB(self) :
        L,Q,indexMap = LQ(self.B)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,1,0]),2)
        newB = np.dot(L,Q)
        self.assertTrue(np.allclose(np.vstack((self.B,[0,1,0],)), newB))
        
    
    def testCFullRoundTrip(self) :
        """ 
        start with the full contstraint set.  
        delete the constraint
        add it back
        then verify we end up with what we started with
        """
        L,Q,indexMap = LQ(self.C)
        L,Q =_del_constraint_LQ(L,Q,indexMap,1)
        L,Q =_del_constraint_LQ(L,Q,indexMap,0)
        L,Q  = _add_constraint_LQ(L,Q,indexMap,np.array([1,0]),0)
        L,Q  = _add_constraint_LQ(L,Q,indexMap,np.array([0,1]),1)
        newC = np.dot(L,Q)
        self.assertTrue(np.allclose(self.C,newC))
    
    def testAFullRoundTrip(self) :
        L,Q,indexMap = LQ(self.A)
        L,Q = _del_constraint_LQ(L,Q,indexMap,1)
        L,Q = _del_constraint_LQ(L,Q,indexMap,0)
        L,Q = _del_constraint_LQ(L,Q,indexMap,2)
        #order of addition must be precisise to end up with the original L and Q
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([1,0,0]),0)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,1,0]),1)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,0,1]),2)
        newA = np.dot(L,Q)
        self.assertTrue(np.allclose(self.A,newA))
        #Let's try a different order of deletion    
        L,Q = _del_constraint_LQ(L,Q,indexMap,0)
        L,Q = _del_constraint_LQ(L,Q,indexMap,1)
        L,Q = _del_constraint_LQ(L,Q,indexMap,2)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([1,0,0]),0)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,1,0]),1)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,0,1]),2)
        newA = np.dot(L,Q)
        self.assertTrue(np.allclose(self.A,newA))
        #Another order of deletion
        L,Q = _del_constraint_LQ(L,Q,indexMap,2)
        L,Q = _del_constraint_LQ(L,Q,indexMap,1)
        L,Q = _del_constraint_LQ(L,Q,indexMap,0)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([1,0,0]),0)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,1,0]),1)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,0,1]),2)
        newA = np.dot(L,Q)
        self.assertTrue(np.allclose(self.A,newA))

    def testEmptyToA(self) :
        X = np.zeros([0,3]) #start with empty constraint matrix for 3 variables
        L,Q,indexMap = LQ(X)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([1,0,0]),0)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,1,0]),1)
        L,Q = _add_constraint_LQ(L,Q,indexMap,np.array([0,0,1]),2)
        newA = np.dot(L,Q)
        self.assertTrue(np.allclose(self.A,newA))


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
    #TODO: possitble typo in Gill book mentioned in R implementation means we have the wrong sign
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
    if gammas[gammaIx] < 1.0 :
        return (gammaIx,gammas[gammaIx],)
    return (None, 1.0,)

def compute_lagrange_multipliers(X,y,A,b,w,verbose=True) :
    """ 
    There is a clever way to solve this using Householder computations,
    using the QT we have already computed, but for now we'll implement the brute force approach.
    """
    g = np.dot(np.dot(X.T,X),w) - np.dot(y,w)
    return np.linalg.lstsq(np.transpose(A.t), g)
     
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
            pass    
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
        #else : # delete a constraint from the working set
        #    #choose a constraint to be deleted s
        #    #delete s from I_k
        #    continue
    return w

def solve_equality_ls(X,y,C,d) :
    #use Golub and Van Loan Chapter 12 method to solve a least squares problem which has only
    #quality constraints
    Q,R = np.linalg.qr(C) #numpy's qr will throw an error if there is insufficient rank
    p = C.shape[1]
    assert R.shape[1] == C.shape[1]
    w = scipy.linalg.solve_triangular(R[:p,:p,d])
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
    unittest.main()

