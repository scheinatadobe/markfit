""" 
The goal of this module is to wrap some disired high level linear regression
solving functionality around third party libraries.

The functionality we want is:
1. solving least squares
2. variance computation for parameters
3. stepwise regression

The approach: wrappers around third party numerical routines such as the QR decomposition.

"""


import numpy as np
import scipy.linalg as linalg
import math
from sys import stderr, stdout
from scipy.linalg.lapack import dpotri #useful for computing standard errors.

def solve(X,y,tolerance=1e-6) :
    """ 
    Solve the regression problem ||Xw - y||^2 using the QR factorization of X.
    Returns: beta, variance estimates for parameters, SSE
    TODO: use an implementation that uses ||Q'Xw - Q'y||^2 to avoid returning Q when it is not
    needed. Somwhere in Lapack there is probably something we could use.
    """
    n = X.shape[0]
    p = X.shape[1]
   
    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    R_diag = np.diag(R)
    rank = np.where(np.abs(R_diag) < tolerance)[0]
    rank = R_diag.shape[0] if rank.shape[0] == 0 else rank[0]
    b     = np.dot(y,Q)[:rank]
    beta  = linalg.solve_triangular(R[:rank,:rank],b,lower=False)
    if rank < R.shape[0] : beta = np.append(beta,np.zeros(R_diag.shape[0] - rank))
    beta[permutation]  = beta.copy() #non-atomic operation requires copy
    R_small = R[:rank,:rank]
    #Computing the inverse from R in a numericaly stable way requires a Cholesky
    #based approach.  I discovered this while testing against statsmodel's 
    #Langley data set which has very high condition number
    XXn1,_   = dpotri(R_small)
    XXn1_diag = np.append(np.diag(XXn1),np.zeros(R.shape[1] - R_small.shape[1]))
    XXn1_diag[permutation] = XXn1_diag.copy()
    betaSigmaSq = XXn1_diag
    resid = y - np.dot(X,beta)
    SSE   = np.dot(resid,resid)
    return beta, betaSigmaSq, SSE, rank,Q

class StepwiseInitError(Exception) : 
    """ Trying to perform stepwise regression with rank deficient base set of predictors"""
    pass
class StepwiseForwardError(Exception) : pass
class StepwiseBackwardError(Exception) : pass

def solveStepwiseInit(X,y,active,tolerance=1e-6) :
    """
    Stepwise regression implementation requires we solve for the lower bound of the 
    search space.
    So how is this different from the non-stepwise case?  
    
    1. Treatment of rank deficiency
        To treat rank deficiency, discover it using rank-revealing QR and then flag 
        deficient columns for non-inclusion in a separate pass of full rank QR
    2. Creation of an augmented R which has two properties:
        a. columns forming the active set form an upper triangular R matrix
        b. the remaining columns have been multiplied by the set of householder matrices 
        sufficient to transform those same active set columns into upper triangular form
    
     
    """
    n = X.shape[0]
    p = X.shape[1]
    #We start the computation with a rank-revealing QR (Scipy's QR routine)  
    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    R_diag = np.diag(R)
    rank = np.where(R_diag < tolerance)[0]
    rank = R_diag.shape[0] if rank.shape[0] == 0 else rank[0]
    allowed_active = np.zeros(active.shape,dtype=np.bool)
    #We use allowed_active to filter out starting set variables that are linear combinations
    #of other starting variables
    allowed_active[permutation[:rank]] = True
    #Having completed this analysis, we will want an un-permuted full-rank QR decopomposition. 
    #For use in the stepwise procedure. Numpy's QR is a full rank variant and that is the one 
    #we apply below (in contrast to scipy's used in the non-stepwise case).
    df  = np.where(allowed_active)[0].shape[0]
    Q,R = np.linalg.qr(X[:,allowed_active],mode='full') #numpy's method differs from scipy's!
    b   = np.dot(Q.T,y)
    beta  = linalg.solve_triangular(R,b,lower=False)
    XXn1  = dpotri(R)
    var   = np.diag(XXn1)
    resid = y - np.dot(X[:,allowed_active],beta)
    SSE   = np.dot(resid,resid)
    return beta, var, SSE, df, np.dot(Q.T,X)
 

def pickForwardVariable(X,y,active,model) :
    """ 
    Identify the best candidate for forward inclusion.
    This version only works for ungrouped variables.
    Returns: a list of one element which is the index of the variable to add.
    We return a list for compatibility witch function which selects groups
    of variables.
    """
    active     = active.copy()
    if np.all(active) : raise StepwiseForwardError()
    print X[:,active].shape, model.beta.shape
    yHat       = np.dot(X[:,active],model.beta)
    resid      = y - yHat
    ix         = np.where(np.logical_not(active))
    candidates = X[:,np.logical_not(active)]
    metric     = np.abs(np.dot(candidates.T,resid)) #
    sqrt       =  np.sqrt(np.sum(candidates * candidates,axis=0))
    metric    /= sqrt
    winner     = ix[0][np.argmax(metric)]
    return [winner]

def pickBackwardVariable(X,y,active,lower_active,model,column_names) :
    """ 
    See Forward variant.
    """
    paramT = model.summary.paramT
    ix     = np.where(active)[0]                                   
    if ix.shape[0] <= 1 :
        raise StepwiseBackwardError()
    if np.sum(active) < X.shape[0] :
        #When variables are less than the number of obsservations, we can use the t-statistic 
        #to pick which variable to drop
        print "paramT: ", paramT
        print "ix: ", ix
        winner = ix[np.argmin(np.abs(paramT))]
        return [winner]
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
        return [bestIx]

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
        new_active = active.copy()
        assert np.all(active[slce]) or np.all(np.logical_not(active[slce]))
        if np.all(active[slce]) :
            continue
        new_active[slce] = True
        Xp = X[:,new_active]
        beta,XXn1_diag, SSE, rank, Q = solve(Xp,y,tolerance=tolerance)
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
    return bestSlice
     
def delNextVariableGroupBruteForce(X,y,active,lower_active,penaltyFn,tolerance=1e-6) :
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
        beta,XXn1_diag, SSE, rank,Q = solve(Xp,y,tolerance=tolerance)
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
    return bestActive 
    
            
