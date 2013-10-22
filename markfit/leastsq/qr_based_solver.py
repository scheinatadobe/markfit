""" 
The goal of this module is to define the key linear and stepwise regerssion intrisics
using QR decomposition and underlying householder and Gibbens rotations strategies
as the underpinning.
"""

#TODO: Look at BLAS givens rotation computation

import numpy as np
import scipy.linalg as linalg
import math
import householder
from sys import stderr, stdout

def solve(X,y,tolerance=1e-6) :
    """ 
    Solve the regression problem ||Xw - y||^2 using the QR factorization of X.
    Returns: beta, diagonal of (X'X)^-1, SSE
    TODO: use an implementation that uses ||Q'Xw - Q'y||^2 to avoid returning Q when it is not
    needed. Somwhere in Lapack there is probably something we could use.
    TODO: I think there is more efficient or numerically stable way of extracting the parameter variance
    values.
    """
    n = X.shape[0]
    p = X.shape[1]
   
    print >>stderr, "performing qr", n, p
    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    print R
    R_diag = np.diag(R)
    print R_diag
    rank = np.where(np.abs(R_diag) < tolerance)[0]
    print "rank: ", rank, tolerance
    rank = R_diag.shape[0] if rank.shape[0] == 0 else rank[0]
    print rank
    b     = np.dot(y,Q)[:rank]
    #print "P is: ", P
    print >>stderr, "triangular solve"
    print R[:rank,:rank]
    print b
    stdout.flush()
    beta  = linalg.solve_triangular(R[:rank,:rank],b,lower=False)
    if rank < R.shape[0] : beta = np.append(beta,np.zeros(R_diag.shape[0] - rank))
    beta[permutation]  = beta.copy() #non-atomic operation requires copy
    R_small = R[:rank,:rank]
    XXn1  = linalg.pinv(np.dot(np.transpose(R_small),R_small))
    XXn1_diag = np.append(np.diag(XXn1),np.zeros(R.shape[1] - R_small.shape[1]))
    XXn1_diag[permutation] = XXn1_diag.copy()
    resid = y - np.dot(X,beta)
    SSE   = np.dot(resid,resid)
    return beta, XXn1_diag, SSE, rank,Q

class StepwiseInitError(Exception) : 
    """ Trying to perform stepwise regression with rank deficient base set of predictors"""
    pass
    

def solveStepwiseInit(X,y,active,tolerance=1e-6) :
    """
    Stepwise regression implementation requires we solve for the lower bound of the search space.
    So how is this different from the non-stepwise case?  
    
    1. Treatment of rank deficiency
        To treat rank deficiency, discover it using rank-revealing QR and then flag 
        deficient columns for non-inclusion in a separate pass of full rank QR
    2. Creation of an augmented R which has two properties:
        a. columns forming the active set form an upper triangular R matrix
        b. the remaining columns have been multiplied by the set of householder matrices sufficient
        to transform those same active set columns into upper triangular form
    
     
    """
    n = X.shape[0]
    p = X.shape[1]
    #We start the computation with a rank-revealing QR (Scipy's QR routine)  
    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    R_diag = np.diag(R)
    rank = np.where(R_diag < tolerance)[0]
    rank = R_diag.shape[0] if rank.shape[0] == 0 else rank[0]
    allowed_active = np.zeros(active.shape,dtype=np.bool)
    #We use allowed_active to filter out starting set variables that are linear combinations of other starting variables
    allowed_active[permutation[:rank]] = True
    #Having completed this analysis, we will want an un-permuted full-rank QR decopomposition. For use in the
    #stepwise procedure. Numpy's QR is a full rank variant and that is the one we apply below 
    #(in contrast to scipy's above).
    df  = np.where(allowed_active)[0].shape[0]
    Q,R = np.linalg.qr(X[:,allowed_active],mode='full') #numpy's method differs from scipy's!
    b   = np.dot(Q.T,y)
    print "shapes: ", b.shape, y.shape, Q.T.shape 
    beta  = linalg.solve_triangular(R,b,lower=False)
    RRn1  = linalg.pinv(np.dot(np.transpose(R),R))
    var   = np.diag(RRn1)
    resid = y - np.dot(X[:,allowed_active],beta)
    SSE   = np.dot(resid,resid)
  
    print "b: ", b, "SSE: ", SSE
    
    return beta, var, SSE, df, np.dot(Q.T,X)

    




    return beta, XXn1_diag[active], SSE, R

def addNextVariable(X,y,active,model,tolerance=1e-6) :
    """ 
    DELETE? trying pick instead of add
    
    Add next variable by determining which variable has the greatest correlation with the 
    residual and adding it to the model.
    This is very efficient, but only works if no variables are grouped (e.g. contrasts)
    """
    active     = active.copy()
    yHat       = np.dot(X[:,active],model.beta)
    resid      = y - yHat
    ix         = np.where(np.logical_not(active))
    candidates = X[:,np.logical_not(active)]
    metric     = np.abs(np.dot(candidates.T,resid)) #
    sqrt       =  np.sqrt(np.sum(candidates * candidates,axis=0))
    metric    /= sqrt
    winner     = ix[0][np.argmax(metric)]
    active[winner] = True
    beta,XXn1_diag, SSE, rank, Q = solve(X[:,active],y,tolerance=tolerance)
    return beta, XXn1_diag, SSE, rank, active

def pickForwardVariable(X,y,active,model) :
    """ 
    Pick next variable by determining which variable has the greatest correlation with the 
    residual and adding it to the model.
    This is very efficient, but only works if no variables are grouped (e.g. contrasts)
    """
    active     = active.copy()
    if np.all(active) : return np.array([],dtype=np.uint64)
    print X[:,active].shape, model.beta.shape
    yHat       = np.dot(X[:,active],model.beta)
    resid      = y - yHat
    ix         = np.where(np.logical_not(active))
    candidates = X[:,np.logical_not(active)]
    metric     = np.abs(np.dot(candidates.T,resid)) #
    sqrt       =  np.sqrt(np.sum(candidates * candidates,axis=0))
    metric    /= sqrt
    winner     = ix[0][np.argmax(metric)]
    result = active.copy()
    result[winner] = True
    return result
    #return (np.array([winner],dtype=np.uint64),)

def pickBackwardVariable(X,y,active,lower_active,model,column_names) :
    """ 
    Pick which variable to drop from the model.
    This function only works when variables are ungrouped (e.g. no constrasts)

    revMap - maps parameter index in current model with X's columns
    """
    print "reverse!"
    paramT = model.summary.paramT
    result = active.copy()
    allowed = np.where(np.logical_not(lower_active))[0]                                   
    if allowed.shape[0] <= 1 :
        #refuse to delete last feature
        return (np.array([],dtype=np.uint64),)
    result = active.copy()
    if np.sum(active) < X.shape[0] :
        #When variables are less than the number of obsservations, we can use the t-statistic 
        #to pick which variable to drop
        winner = allowed[np.argmin(np.abs(paramT[allowed]))]
        print "winner is: ", column_names[winner]
        result[winner] = False
    else :
        #we need to use brute force to figure out which variable to remove
        revMap = np.zeros([np.sum(self.active)],dtype=np.uint64)
        j = 0 # build a reverse map from currently used variables to original set
        for i,v in enumerate(active) :
            if v : 
                revMap[j] = i
                j += 1
        bestSSE = None
        bestIx  = None
        for i in revMap :
            new_active = active.copy()
            new_active[i] = False
            Xp = X[:,new_active]
            beta,XXn1_diag, SSE, rank = solve(Xp,y,tolerance=tolerance)
            if bestSSE is None or bestSSE > SSE :
                bestIx = i
                besetSSE = SSE
        assert bestIx is not None
        result[bestIx] = False
    return result
    

   
def pickNextVariableGroupBruteForce(X,y,active,term_slices,penaltyFn,tolerance=1e-6) :
    """ This version treats categorical variables as a single entity for edition or deletion.
        Unfortunately, the only way I know to solve this is brute force for now. I don't like it,
        but it serves the purpose of R compatibility. """
    bestSlice = None
    bestSSE   = None
    bestBeta  = None
    bestXXn1_diag = None
    bestRank  = None
    bestActive = None
    bestFit   = None
    for term, slce in term_slices.iteritems() :
        print term,slce
        new_active = active.copy()
        assert np.all(active[slce]) or np.all(np.logical_not(active[slce]))
        if np.all(active[slce]) :
            continue
        new_active[slce] = True
        Xp = X[:,new_active]
        beta,XXn1_diag, SSE, rank = solve(Xp,y,tolerance=tolerance)
        fitness = penaltyFn.getValue(X.shape[0],rank,SSE)
        print "fitness: ", fitness
        if bestSlice is None or fitness > bestFit :
            bestFit   = fitness
            bestSlice = slce
            bestSSE   = SSE
            bestBeta  = beta
            bestXXn1_diag = XXn1_diag
            bestRank  = rank
            bestActive = new_active
    assert bestRank == np.sum(bestActive), (bestRank,bestActive)
    assert bestSlice is not None
    return bestActive
     
def delNextVariableGroupBruteForce(X,y,active,penaltyFn,tolerance=1e-6) :
    """ This version treats categorical variables as a single entity for edition or deletion.
        Unfortunately, the only way I know to solve this is brute force for now. I don't like it,
        but it serves the purpose of R compatibility. """
    bestSlice = None
    bestSSE   = None
    bestBeta  = None
    bestXXn1_diag = None
    bestRank  = None
    bestActive = None
    bestFit   = None
    for term, slce in X.design_info.term_slices.items() :
        new_active = active.copy()
        assert np.all(active[slce]) or np.all(np.logical_not(active[slce]))
        if np.all(np.logical_not(active[slce])) :
            continue
        new_active[slce] = False
        Xp = X[:,new_active]
        beta,XXn1_diag, SSE, rank = solve(Xp,y,tolerance=tolerance)
        fitness = penaltyFn.getValue(X.shape[0],rank,SSE)
        if bestSlice is None or fitness > bestFit :
            bestFit   = fitness
            bestSlice = slce
            bestSSE   = SSE
            bestBeta  = beta
            bestXXn1_diag = XXn1_diag
            bestRank  = rank
            bestActive = new_active
    assert bestRank == np.sum(bestActive), (bestRank,bestActive)
    assert bestSlice is not None
    return bestBeta,bestXXn1_diag,bestSSE,bestRank,bestActive
    
            
