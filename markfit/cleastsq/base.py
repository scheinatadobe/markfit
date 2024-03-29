
#to test:
#python -m markfit.cleastsq.base

#if __name__ == "__main__" and __package__ is None:
#    #for debugging
#    from os import sys, path
#    #sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from ..numerics import householder as hh
from ..numerics import givens
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
    assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)
    y = y.reshape(y.shape[0])
    print "let's check some dimensions: np.dot(X,Z) ", np.dot(X,Z).shape
    print "np.dot(X,w) ", np.dot(X,w).shape
    print "y: ", y.shape
    print "np.dot(X,w) - y)", np.dot(X,w) - y
    print "newX for ls: ", np.dot(X,Z)
    print "lsqsq result: ", np.linalg.lstsq(np.dot(X,Z),np.dot(X,w) - y)
    res,_,_,_ = np.linalg.lstsq(np.dot(X,Z),np.dot(X,w) - y)
    
    print "after Z it is: ", np.dot(Z,res)
    #res /= np.linalg.norm(res)
    print "after norm: ", res
     
    return res

def compute_step_length(X,y,w,Z,A,b,active,p_z) :
    """
    take best step in direction p_z
    If taking p_z does not break any constraints, just go with it.
    Otherwise, take the largest distance in direction p_z such that no constraints are broken.
    Return index of constraint chosen (possibly None), along with the step length
    """
    
    #TODO: can I unit test this
    #TODO: double check that 1.0 is the optimimum in this direction (chk normalization of direction)
    length        = 1.0
    assert(length >= 1e-8)
    direction     = p_z / length
    inactive      = np.logical_not(active)
    constrainable = np.logical_and(inactive, np.dot(A[inactive],p_z) < 0 )
    ix = np.where(constrainable)
    if not np.any(inactive) : 
        return None,length,
    gammas = list((b[constrainable] - np.dot(A[constrainable],w))/ np.dot(A[constrainable], \
                        direction))
    gammaIx = np.argmin(gammas)
    if gammas[gammaIx] < 1.0 :
        return (gammaIx,gammas[gammaIx],)
    return (None, 1.0,)

def compute_lagrange_multipliers(Ahat,g) :
    """ 
    There is a clever way to solve this using Householder computations,
    using the QT we have already computed, but for now we'll implement the brute force approach.
    """
    #g = np.dot(np.dot(X.T,X),w) - np.dot(y,w)
    #print "A.shape: ", A.shape, "g.shape: ", g.shape
    result,_,_,_ = np.linalg.lstsq(Ahat, g)
    return result
     
def solve_inequality_ls(X,y,A,b,alwaysActive,verbose=True) :
    """
    Minimize || Xw - y||^2 
    s.t.  Aw >= b
    and   A[alwaysActive]w = b
    The nature of the second constraint set is that equality constraints have been folded into
    the A matrix, but the indices of those constraints must always be in the working set.
    aIx are those constraints that must always be active; these are equality constraints from the
    original problem.
    """
    #The idea behind the implementation is to ensure the candidate solution always satisfies
    #the active constraints.  When we add a constraint it is because the natural step length 
    #to the least squares problem defined by the current active set would cause
    #us to break the constraint... so we add the constraint and shorten the step accordingly.
    #when we remove the constraint, this is true by default.  Below, we compute a solution which
    #obeys quality for all constraints.  Don't worry, we try to remove a constraint in the
    #first iteration of the loop.
    #TODO: Could I come up with a feasible solution and initizlize the alg from there instead?
    i = 0
    active         = np.ones(A.shape[0],dtype=np.bool)
    L, Q, indexMap = LQ(A) # start with all constraints active.
    print "L:", L
    print "Q:", Q
    print "indexMap: ", indexMap
    print "L: ", L.shape, "Q:", Q.shape, "len: ", len(indexMap), "A: ", A, A.shape
    Z            = Q[:,:A.shape[0]]
    w            = solve_equality_ls(X,y,A,b)
    print "pre initial solution: ", w
    p_k          = find_feasible_direction(X,y,w,Z)
    #j, alpha   = compute_step_length(X,y,w,Z,A,b,active,p_k)
    #w          += alpha * p_k

    print "initial solution: ", w

    while i < np.max(X.shape) and len(indexMap.back) : #not has_converged(X,y,A,b,w,active,g,Z) :
        print "LOOPING! ", w
        #loop invariant 1:  we have already solved optimimum for the current working set. 
        #loop invariant 2:  if is no constraint to drop, then our optimization work is done
        i += 1 #just for debugging
        #The lagrange multipliers tell us if there is any constraint to drop
        lambdas = compute_lagrange_multipliers(np.dot(A,Z),p_k)
        print "returned lambdas: ", lambdas
        j = np.argmin(lambdas)
        if lambdas[j] >= 0 : # no constraint to drop?  All done!
            return w
        print "About to delete: ", j, Q.shape, L.shape
        print "L:", L
        print "Q:", Q
        print "indexMap.back : ", indexMap.back
        print "j: ", j
        L,Q = _del_constraint_LQ(L,Q,indexMap,j)
        p_k = find_feasible_direction(X,y,w,Z)
        print "p_k: ", p_k
        j, alpha   = compute_step_length(X,y,w,Z,A,b,active,p_k)      
        print "j, alpha: ", j, alpha
        w -= alpha * p_k
        if j is not None :
            #add a constraint and then continue
            #how do I map the j to something else that gets passed?
            L,Q = _add_constraint_LQ(L,Q,A[j],j)
    if i >= np.max(X.shape) : 
        assert "solve_inquality_ls failed to converge!"
    return w

def solve_equality_ls(X,y,C,d) :
    #use Golub and Van Loan Chapter 12 method to solve a least squares problem which has only
    #quality constraints
    assert C.shape[1] == X.shape[1]
    assert d.shape[0] == C.shape[0]
    assert X.shape[0] == y.shape[0]
    print "solving with constrints : ", C, " = ", d
    Q,R = np.linalg.qr(C) #numpy's qr will throw an error if there is insufficient rank
    p = C.shape[1]
    assert R.shape[1] == C.shape[1]
    w = scipy.linalg.solve_triangular(R[:p,:p],d)
    if p >= X.shape[1] : return w
    X = np.dot(X,Q)
    
    print "p: ", p, "X: ", X, "y: ", y, "w: ", w
    print "let's check dimensions: ", (X[:,p:]).shape, (y - np.dot(X[:,:p],w)).shape
    z = np.linalg.lstsq(X[:,p:], y - np.dot(X[:,:p],w))
    return np.dot(Q[:,:p],w) + np.dot(Q[:,p+1:],z)

def ls(X,y,A,b,C,d,w0,verbose=True) :
    """
    Minimize || Xw - y||^2 
    s.t.  Aw >= b
    and   Cw  = d.
    
    w0 is a given feasible solution, which is used as a starting point.
    If A,b,C,d may be passed as None when there are no applicable constraints.
    """
    #TODO: make w0 optional.
    assert (A is not None and b is not None) or (A is None and b is None)
    assert (C is not None and d is not None) or (C is None and d is None)
    
    #The approach we use requries  a feasible starting point, and guarantees that all
    #subsequent updates are also feasible
    assert __is_feasible_point(A,b,C,d,w0,verbose=verbose)
    if A is None and C is None :
        #this is an ordinary least squares problem
        return np.linalg.lstsq(X,y)
    elif A is None :
        #use Golub and Van Loan Chapter 12 method for equality (only) constraints
        return solve_equality_ls(X,y,C,d)
    else :
        aIx = np.zeros(A.shape[0],dtype=np.bool)
        if C is not None :  #add equality constraints to the working set
            aIx.append(np.ones(C.shape[0],dtype=np.bool))
        #with equality constraints encoded as working set, we can merge A and B matrices
        #and pass along to a specialized routine. aIx is passed to indicate these certain
        #constraints must be permanently used in the active set
        newA = A if C is None else np.vstack((A,C,))
        newb = b if d is None else np.vstack((b,d,))
        return solve_inequality_ls(X,y,newA,newb,aIx,verbose=True)

if __name__ == "__main__" : 
    unittest.main()

