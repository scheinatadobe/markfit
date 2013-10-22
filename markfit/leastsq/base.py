#In this module, we write code to handle the more high level interfaces to linear modelling.
#Users provide high level details such as a data frame and formula, and the code
#here dispatches to the appropriate solvers

import numpy as np
import patsy
import pandas
#import sweep_based_solver
import adjust_based_solver
import qr_based_solver
import scipy.stats as stats
import scipy.linalg as linalg
from numpy import allclose
from stepwise_penalties import AIC, AICc
from sys import stdout, stderr
from patsy import EvalEnvironment, ModelDesc, Term
import math
print "test"
run_paranoid_tests = True #some are expensive

#fitting strategies coded as variables:
#QR decomposition
QR="QR"
#Adjust-based operations on X'X (Goodnight, 1979)
Adjust="Adjust"
#Sweep-based operations on X'X (Goodnight, 1979)
Sweep="Sweep"

class LinearModelFit :
    def __init__(self,beta,summary) :
        self.beta    = beta
        self.summary = summary

class LinearModelSummary :
    def __init__(self,n,df,beta,SSTO,SSE,MSE,SSR,MSR,f,pVal,RSq,paramT,paramPVal,S,column_names,
                modelDesc) :
        self.n = n
        self.df = df
        self.beta = beta
        self.SSTO = SSTO
        self.SSE  = SSE
        self.MSE  = MSE
        self.SSR  = SSR
        self.MSR  = MSR
        self.f    = f
        self.pVal = pVal
        self.RSq  = RSq
        self.paramT = paramT
        self.paramPVal = paramPVal
        self.column_names = column_names
        self.S         = S
        self.modelDesc   = modelDesc

    def write(self,stdout=stdout,pad=4) :
        
        lines = []
        print >>stdout, "Coefficients:"
        maxlength = max(map(lambda x : len(x), self.column_names))
        first= "%-" + str(maxlength) + "s" 
        print >>stdout, first%"" + " %12s %+11s %+11s %+9s"%("Estimate","Std. Error","t value","Pr(>|t|)")
        for i in range(self.beta.shape[0]) :
            val  = self.beta[i]
            name = self.column_names[i]
            labfmt = "%-" + str(maxlength) + "s"
            print >>stdout, labfmt % name,
            print >>stdout," %11.2f %11.2f %11.3f %9.4g"%(val,self.S[i],self.paramT[i],\
                                                            self.paramPVal[i])
        pValueText = "<" + "%5.4g"%(np.finfo(np.double).eps)
        if self.pVal >= np.finfo(np.double).eps :
            pValueText = "%6.4g"%self.pVal  
        lines.append("") 
        lines.append("ANOVA Table")
        lines.append("-----------")
        lines.append("             %+9s %+12s %9s" % ("Sum","df","Mean"))
        if self.df > 1 :
            lines.append("SSR          %8.4g %12d %8.4g"%(self.SSR,self.df-1,self.MSR))
        lines.append( "SSR          {:>9.4g} {:>12d} {:>9.4g}".format(self.SSE,self.n - self.df,self.MSE))
        #lines.append("SSE          %8.4g %12d %8.4g"%(self.SSE,self.n-self.df,self.MSE))
        lines.append("")
        if self.df > 1 : 
            lines.append("F-statistic: %d"%int(self.f) + \
            " on %d and %d degrees of freedom"%(self.df-1,self.n - self.df) + \
            ", p-value: %s"%pValueText)
        lines.append("Multiple R Squared: %4.3f"%self.RSq)
        lines.append("")
        print >>stdout, "\n".join(lines)     

def computeSummary(beta,betaSigmaSq,SSE,SSTO,n,df,column_names,formula) :
    """
    Summaryize the linear regression model
    #y    - the dependent variable
    #X    - the design matrix
    A    - the augmented matrix after solving
    n    - number of observations
    df   - degrees of freedom which differs from predictors when features are co-linear
    SSTO - the total sum of squares (can be computed without running a regression)
    returns a summary object
    
    The design of this summary is to not use X or y directly.  In this manner it might be used
    in a map-reduce context where N is too large to fit X or y in memory.

    """
    
    assert n>df
    assert df>0
    MSE        =   SSE / (n - df)
    SSR        =   SSTO - SSE #fundamental property of linear regression     
    MSR        =   SSR/(df - 1)
    f_stat     =   MSR/MSE #TODO: sensible checking
    pVal       =   1 - stats.f.cdf(f_stat,df-1,n-df)
    RSq        =   SSR / SSTO
    #On to estimation of parameter-specific values
    #for now hard code single response
    S  = np.sqrt(betaSigmaSq * MSE) #estimate parameter sigma
    paramT     = beta / S
    paramT[S == 0] = 0
    paramPVal  = 2 * (1 - stats.t.cdf(np.abs(paramT), n - df))
    return LinearModelSummary(n,df,beta,SSTO,SSE,MSE,SSR,MSR,f_stat,pVal,RSq,paramT,paramPVal,S,
                                column_names,formula)

def lm(formula,dataFrame,fitStrategy="QR",tolerance=1e-6) :

    """ 
    formula     - a patsy formula
    dataFrame   - a pandas data frame
    fitStrategy - the algorithm used to fit the model. 'QR' is recommended for maximum 
                  numerical stability
    tolerance   - a tolerance parameter used to identify near singularity in the model fitting
                  strategy
    """
    y,X = patsy.dmatrices(formula,dataFrame)
    assert y.shape[1] == 1, "Multiple dependent variables are not yet supported."
    y = y.flatten()
    n   = y.shape[0]
    p   = X.shape[1]
     
    XX  = np.dot(X.T,X)
    #print X.design_info.term_names    
    #print y.shape
    
    if fitStrategy == QR :
        beta,betaSigmaSq,SSE,df,Q = qr_based_solver.solve(X,y,tolerance=tolerance)
    elif fitStrategy == "Adjust" or fitStrategy == "Sweep" :
    	Xy  = np.array([np.dot(X.T,y)])
    	A = np.append(XX,Xy.T,axis=1)
    	if fitStrategy == "Adjust" : 
		    A = np.append(A,np.identity(XX.shape[0]),axis=1)
    	yy = np.array([[np.dot(y,y)]])
    	bottom = np.append(Xy,yy.T,1)
    	if fitStrategy == "Adjust" : 
		    bottom = np.append(bottom,np.zeros([1,p]),1)
    	A = np.append(A,bottom,0)
        if fitStrategy == "Adjust" : 
            adjust_based_solver.solve(A)
            XXn1 = A[:p,p+1:]
        else :
            sweep_based_solver.solve(A)
            XXn1 = A[:p,:p]
        betaSigmaSq = np.diag(XXn1)
        beta = A[:p,p]
        SSE  = A[p,p]
    else : 
        raise "Invalid fit strategy selected: '" + fitStrategy + "'"
    
    #The problem is solved.  On to computing ANOVA table and other diagnostics! 
    residWithMean = y - np.mean(y)
    SSTO = np.dot(residWithMean,residWithMean)
    env = EvalEnvironment.capture() 
    modelDesc = ModelDesc.from_formula(formula,env)
    summary = computeSummary(beta,betaSigmaSq,SSE,SSTO,n,df,X.design_info.column_names,modelDesc)
    #print summary.beta
    return LinearModelFit(summary.beta,summary)

class StepwiseNoAction(Exception): 
    #exception to throw no variables meet the stepwise search criteria
    pass

def AIC(n,df,SSE) :
    return -df - n * math.log(SSE/n)
    
class StepwiseFitter :
    def __init__(self,
        currentFit,
        X,
        y,
        X_columns,
        y_columns,
        active,       #currently active variables
        lower_active, #required variables
        penaltyFn,
        tolerance=1e-6,
                trace=False,traceFile=stdout,groupVars=False) :
        """ groupFactors: treat the multiple columns generated by categorical variables as 
            a single unit for inclusion or deletion. """
        ### TODO: generalize the penalty fn interface
        self.currentFit = currentFit
        self.active     = active
        self.X          = X
        self.X_columns  = X_columns
        self.y_columns  = y_columns
        self.y          = y
        self.tolerance  = tolerance
        self.__SSTO     = currentFit.summary.SSTO #this will not change, so let's permanently track
        self.penaltyFn  = penaltyFn
        self.trace      = trace
        self.traceFile  = traceFile
        self.groupVars  =  groupVars
        self.stepCount  = 0
        self.lower_active = lower_active

    def step1(self,direction) :
        """
        Add a single predictor to the model.
        raise StepwiseNoAction() if there is no predictor which increases the 
        penalized likelihood function.
        """
         
        print "direction: ", direction 
        print "lower_active: ", self.lower_active
        assert direction in set(["forward","backward"]), \
            "direction must be either 'forward' or 'backward.' Passed: '" + direction + "'"
        old_summary = self.currentFit.summary
        print "checking conditions"
        if direction == "forward" and np.all(self.active) : 
            raise StepwiseNoAction("All features are already added.")
        if direction == "backward" and np.all(self.active == self.lower_active)  :
            raise StepwiseNoAction("No features can be removoved")
        X = self.X
        y = self.y.flatten()
        #revMap = np.zeros([np.sum(self.active)],dtype=np.uint64)
        #j = 0 # build a reverse map from currently used variables to original set
        #for i,v in enumerate(active) :
        #    if v : 
        #        revMap[j] = i
        #        j += 1
        #
        #column_names = X.design_info.column_names
        print "entering case: "
        if not self.groupVars and direction == "forward" :   
            active = qr_based_solver.pickForwardVariable(X,y,self.active,self.currentFit)
        elif direction=="forward" :
            active = qr_based_solver.pickNextVariableGroupBruteForce(X,y,self.active,
                                                X.design_info.term_slices,
                                                self.penaltyFn,
                                                tolerance=self.tolerance)
        elif not self.groupVars and direction == "backward" :
            print "entering pickBackwardVariable..."
            active = qr_based_solver.pickBackwardVariable(X,y,self.active,self.lower_active,
                                                            self.currentFit,self.X_columns)
        
        else : 
            raise Error("Not implemented yet!")
        
        if np.all(active == self.active) :
            print "They are all the same!"
            raise StepwiseNoAction()
        beta, var, SSE, df = qr_based_solver.solve(X[:,active],y,self.tolerance) 
        
        column_names = [ name for i,name in enumerate(self.X_columns) if active[i]  ]
        
        if self.penaltyFn.getValue(X.shape[0],df,SSE) <= \
                self.penaltyFn.getValue(X.shape[0],old_summary.df,old_summary.SSE):
            print "old value", self.penaltyFn.getValue(X.shape[0],old_summary.df,old_summary.SSE)
            print "new value", self.penaltyFn.getValue(X.shape[0],df,SSE)
            raise StepwiseNoAction()
        modelDesc = ModelDesc(self.y_columns,column_names)
        summary = computeSummary(beta,var,SSE,self.__SSTO,X.shape[0],df,
                                column_names,modelDesc)
        
        if self.trace : 
            print >>self.traceFile, "Step Count: ", self.stepCount
            summary.write(self.traceFile)
        self.active  = active
        self.currentFit = LinearModelFit(summary.beta,summary)
        self.stepCount += 1

    def stepBack(self) :
        """
        Remove a single predictor.
        raise StepwiseNoAction() if there is no predictor which can be removed while
        increasing the penalized likelihood.
        """
        old_summary = self.currentFit.summary
        active = self.active[:]
        if np.all(active) : raise StepwiseNoAction("All features are already added.")
        X = self.X
        y = self.y.flatten()
        #column_names = X.design_info.column_names
        if not self.groupVars :
            if old_summary.df < old_summary.n : #we can use the t-statistic
                i = np.argmin(np.abs(old_summary.paramT))
            else : raise "Not Implemented yet!"
            active[i] = False 
            beta,betaSigmaSq,SSE,df = qr_based_solver.solveWithActive(X,y,active,tolerance=self.tolerance)
        else :
                
            beta,betaSigmaSq,SSE,df,active = qr_based_solver.delNextVariableGroup(X,y,active,AIC,
                                                tolerance=self.tolerance)
        column_names = [ name for i,name in enumerate(self.X_columns) if active[i]  ]
        
        if self.penaltyFn.getValue(X.shape[0],df,SSE) <= \
                self.penaltyFn.getValue(X.shape[0],old_summary.df,old_summary.SSE):
            raise StepwiseNoAction()
        modelDesc = ModelDesc(self.y_columns,column_names)
        summary = computeSummary(beta,betaSigmaSq,SSE,self.__SSTO,X.shape[0],df,
                                X.design_info.column_names,modelDesc)
        if self.trace : 
            print >>self.traceFile, "Step Count: ", self.stepCount
            summary.write(self.traceFile)
        self.active  = active
        self.currentFit = LinearModelFit(summary.beta,summary)
        self.stepCount += 1
 

    def step(self,direction="both") :
        if self.trace :
            print >>self.traceFile, "Step Count: ", self.stepCount
            self.currentFit.summary.write(self.traceFile)
        if direction == "forward" :
            try :
                while 1 :
                    self.step1("forward")
            except StepwiseNoAction, e : pass
        elif direction == "backward" :
            try :
                while 1 :
                    self.step1("backward")
            except StepwiseNoAction, e : pass
        elif direction == "both" : # a heuristic based on alternating between adding and deleting features
            while 1 :
                changes = 0
                try :
                    while 1 :
                        self.step1("forward")
                        changes += 1
                except StepwiseNoAction, e : pass
                try :
                    while 1 :
                        self.step1("backward")
                        changes += 1
                except StepwiseNoAction, e : pass
                if not changes : break

        else : raise StepwiseError("direction argument should be one of: 'forward', 'backward', or 'both'")
        


def stepwiseInit(upperScope,
                    dataFrame,
                    lowerScope=None,
                    startScope=None,
                    trace=False,
                    traceFile=stdout,
                    groupVars=False,
                    penaltyFn=AICc()) :
    #todo: add lower scope
    ### The first thing that has to happen is establish the starting set of variables
    ### this will be stored in boolean vector active
    env = EvalEnvironment.capture()     
    upperScopeDesc = ModelDesc.from_formula(upperScope, env)
    startScopeDesc = None if startScope is None else ModelDesc.from_formula(startScope,env)
    lowerScopeDesc = None if lowerScope is None else ModelDesc.from_formula(lowerScope,env)
    if not lowerScope and Term([]) not in upperScopeDesc.rhs_termlist :
        raise StepwiseError("A lower scope of the model search must be specified when " + \
        "the upperScope does not contain an intercept")
    if not lowerScope : #build a formula with only an intercept
        lowerScopeDesc = ModelDesc(upperScopeDesc.lhs_termlist,[Term([])])
        lowerScope     = lowerScopeDesc.describe()
    if not startScope :
        startScopeDesc = lowerScopeDesc
        startScope     = lowerScopeDesc.describe()
    #TODO: check that lower scope is consistant with upper scope
    #TODO: check that startingscope is consistent with lower and upper scopes
    rhs_set = set(upperScopeDesc.rhs_termlist)
    for item in lowerScopeDesc.rhs_termlist :
        if item not in rhs_set : raise StepWiseError("term " + item + " from formula:\n" + \
            lowerScope + "\nnot found in:\n" + \
            upperScope)
    for item in startScopeDesc.rhs_termlist :
        if item not in rhs_set : 
            raise StepWiseError("term " + item + " from formula:\n" + \
                        startScope + "\nnot found in:\n" + \
                                    upperScope)

    y,X      = patsy.dmatrices(upperScope, data=dataFrame)
    y,Xprime = patsy.dmatrices(startScope, data=dataFrame)
    y,Xlower = patsy.dmatrices(lowerScope, data=dataFrame) 
    active   = np.zeros(X.shape[1],dtype=np.bool)
    lower_active = active.copy()
    lowerMsk = active.copy()
    assert y.shape[1] == 1, "Multiple responses not yet supported."
    y = y.flatten()
    featMap = dict([(name,index) for index,name in enumerate(X.design_info.column_names)]) 
    for feat in Xprime.design_info.column_names :
        active[featMap[feat]] = True
    for feat in Xlower.design_info.column_names :
        lower_active[featMap[feat]] = True
    #next step: fit model using only the active set of features
    print >>stderr, "about to solve"
    beta, betaSigmaSq, SSE, df, Q = qr_based_solver.solve(X[:,active],y.flatten())
    print >>stderr, "solved"

    residWithMean = y - np.mean(y)
    SSTO = np.dot(residWithMean,residWithMean)
    summary = computeSummary(beta,betaSigmaSq,SSE,SSTO,X.shape[0],df,
                                Xprime.design_info.column_names,startScope)
    
    #print "initizlizing with: ", summary.write(stdout)
    return StepwiseFitter(LinearModelFit(summary.beta,summary),X,y,X.design_info.column_names,
                            upperScopeDesc.lhs_termlist,active,lower_active,penaltyFn,trace=trace,
                            traceFile=traceFile, groupVars=groupVars)


if __name__ == "__main__" :
    
    data = pandas.io.parsers.read_csv("salary2.txt")

    
    #fit = lm("sl ~ 1+sx+rk+yr+dg+yd+dg2",data,fitStrategy=QR)
    #fit.summary.write()
#    fit = lm("sl ~ 1+sx+rk+yr+dg+yd+yd2",data,fitStrategy=Adjust)
#
#    #fit = lm("sl ~  dg + yd +yd2", data)
#
#    fit.summary.write()
#
    #data = pandas.io.parsers.read_csv("salary2.txt")
    #stepper = stepwiseInit("sl ~ sx+rk+yr+dg+yd",data,trace=True,groupVars=False)
    #stepper.step(direction="forward")
    stepper = stepwiseInit("sl ~ sx+rk+yr+dg+yd",data,startScope="sl ~ sx+rk+yr+dg+yd",
    trace=True,groupVars=False)
    #stepper.step(direction="backward")
      

