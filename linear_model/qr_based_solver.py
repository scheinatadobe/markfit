import numpy as np
import scipy.linalg as linalg
import math

def solve(X,y,tolerance=1e-6) :
    """ 
    Solve the regression problem ||Xw - y||^2 using the QR factorization of X.
    Returns: beta, diagonal of (X'X)^-1, SSE
    TODO: use an implementation that uses ||Q'Xw - Q'y||^2 to avoid returning Q when it is not
    needed.
    TODO: optionally return Q and R for incremental updating.  May require augmenting scipy
    if this functionality is not there.
    """
    
    n = X.shape[0]
    p = X.shape[1]
    Q,R,permutation = linalg.qr(X,pivoting=True,mode='economic')
    R_diag = np.diag(R)
    rank = np.where(R_diag < tolerance)[0]
    rank = R_diag.shape[0] if rank.shape[0] == 0 else rank[0]
    b     = np.dot(y,Q)[:rank]
    #print "P is: ", P
    beta  = linalg.solve_triangular(R[:rank,:rank],b,lower=False)
    if rank < R.shape[0] : beta = np.append(beta,np.zeros(R_diag.shape[0] - rank))
    beta[permutation]  = beta.copy() #non-atomic operation requires copy
    R_small = R[:rank,:rank]
    XXn1  = linalg.pinv(np.dot(np.transpose(R_small),R_small))
    XXn1_diag = np.append(np.diag(XXn1),np.zeros(R.shape[1] - R_small.shape[1]))
    XXn1_diag[permutation] = XXn1_diag.copy()
    resid = y - np.dot(X,beta)
    SSE   = np.dot(resid,resid)
    return beta, XXn1_diag, SSE, rank

def solveWithActive(X,y,active,tolerance=1e-6) :
    #Like the ordinary solver, but uses the active mask to determine which columns
    #to use
    return solve(X[:,active],y,tolerance=1e-6)

def addNextVariable(X,y,active,model,tolerance=1e-6) :
    """ Measuing the normalized correlation with the residual of the current model is the way to determine which 
        variable will decrease the SSE the most if added.
        This is very efficient, but does not handle the case where we want variables associated with a single categorical
        predictor added as a group.  A separate function handles that case
    """
    active     = active.copy()
    yHat       = np.dot(X[:,active],model.beta)
    resid      = y - yHat
    ix         = np.where(np.logical_not(active))
    candidates = X[:,np.logical_not(active)]
    metric     = np.abs(np.dot(candidates.T,resid)) #
    sqrt =  np.sqrt(np.sum(candidates * candidates,axis=0))
    metric /= sqrt
    winner = ix[0][np.argmax(metric)]
    active[winner] = True
    beta,XXn1_diag, SSE, rank = solve(X[:,active],y,tolerance=tolerance)
    return beta,XXn1_diag, SSE, rank, active

    

def addNextVariableGroup(X,y,active,penaltyFn,tolerance=1e-6) :
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
        if np.all(active[slce]) :
            continue
        new_active[slce] = True
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
    
def delNextVariableGroup(X,y,active,penaltyFn,tolerance=1e-6) :
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
    
            
