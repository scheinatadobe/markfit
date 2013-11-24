"""
This module implements high level interfaces for linear modeling. Users provide
a data frame and formula.  The code dispatches to the appropriate solvers.
"""


from markfit.leastsq import adjust_based_solver
from markfit.leastsq import qr_based_solver
from markfit.leastsq.stepwise_penalties import AIC, AICc

import numpy as np
import pandas
import patsy
import scipy.stats as stats
import scipy.linalg as linalg
from sys import stdout, stderr
import math
from prettytable import PrettyTable

#fitting strategies coded as variables:
#QR decomposition
QR = "QR"
#Adjust-based operations on X'X (Goodnight, 1979)
ADJUST = "Adjust"
#Sweep-based operations on X'X (Goodnight, 1979)
SWEEP = "Sweep"

# def floatFormatter(value,precision):
#     if value > 1e6 :
#         return "%g"%value
#     if value > 1e5 :
#         return "%d"%(int(value + 0.5))
#     if value > 1e4 :
#         return  
#     
    
class LinearModelFit :
    """
    Holds the parameters and summary information associated with the
    linear regression model
    """
    def __init__(self, beta, summary) :
        self.beta    = beta
        self.summary = summary

class LinearModelSummary :
    def __init__(self,n,df,beta,SSTO,SSE,MSE,SSR,MSR,f,pVal,RSq,paramT,paramPVal,S,column_names,
                modelDesc) :
        #TODO: encapsolate df calcs so that printers don't have to do this work.
        self.n = n
        self.degrees_freedom = df
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

   
    def textPretty(self):
        """
        This is an experiment to use the PrettyTable module
        """
        #start by building the coefficients table
        def floatFmt(x) :
            #0.7.2 prettytable does not do %g, so we convert to text with the 
            #desired formatting
            return "%6g"%x
        
        p = PrettyTable()
        p.add_column("",self.column_names)
        p.align[""] = 'l'
        p.add_column("Estimate",map(floatFmt,self.beta))
        p.align["Estimate"] = "r"
        p.add_column("Std. Error",map(floatFmt,self.S))
        p.align["Std. Error"] = "r"
        p.add_column("t value",map(floatFmt,self.paramT))
        p.align["t value"] = "r"
        p.add_column("p value",map(floatFmt,self.paramPVal))
        p.align["p value"] = "r"
        
        anova = PrettyTable()
        anova.add_column("",["SSR","SSE"])
        anova.add_column("Sum",map(floatFmt, [self.SSR,self.SSE]))
        anova.align["Sum"] = "r"
        anova.add_column("df", [self.degrees_freedom - 1, self.n - self.degrees_freedom] )
        anova.align["df"] = "r"
        anova.add_column("Mean",map(floatFmt,[self.MSR,self.MSE]))
        anova.align["Mean"] = "r"
        return p,anova
    

    def write(self,stdout=stdout,pad=4) :
        
        lines = []
        print >>stdout, "Coefficients:"
        maxlength = max(map(lambda x : len(x), self.column_names))
        first= "%-" + str(maxlength) + "s" 
        print >>stdout, first%"" + " %13s %+11s %+11s %+11s"%("Estimate",
                                                             "Std. Error",
                                                             "t value",
                                                             "Pr(>|t|)")
        for i in range(self.beta.shape[0]) :
            val  = self.beta[i]
            name = self.column_names[i]
            labfmt = "%-" + str(maxlength) + "s"
            print >>stdout, labfmt % name,
            print >>stdout, "%13.4g %11.3g %11.3g %11.4g"%(val,self.S[i],self.paramT[i],\
                                                            self.paramPVal[i])
        pValueText = "<" + "%5.4g"%(np.finfo(np.double).eps)
        if self.pVal >= np.finfo(np.double).eps :
            pValueText = "%6.4g"%self.pVal  
        lines.append("") 
        lines.append("ANOVA Table")
        lines.append("-----------")
        lines.append("             %+9s %+12s %9s" % ("Sum","df","Mean"))
        if self.degrees_freedom > 1 :
            lines.append("SSR          %8.4g %12d %8.4g"%(self.SSR,self.degrees_freedom-1,self.MSR))
        lines.append( "SSR          {:>9.4g} {:>12d} {:>9.4g}".format(self.SSE,
                                                                      self.n - self.degrees_freedom,
                                                                      self.MSE))
        #lines.append("SSE          %8.4g %12d %8.4g"%(self.SSE,self.n-self.df,self.MSE))
        lines.append("")
        if self.degrees_freedom > 1 : 
            lines.append("F-statistic: %d"%int(self.f) + \
            " on %d and %d degrees of freedom"%(self.degrees_freedom-1,
                                                self.n - self.degrees_freedom) + \
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
    env = patsy.EvalEnvironment.capture() 
    modelDesc = patsy.ModelDesc.from_formula(formula,env)
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
        trace=False,
        traceFile=stdout,
        groupVars=False) :
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
        assert direction in ["forward","backward"] 
        print "step1 direction: ", direction 
        print "lower_active: ", self.lower_active
        print "entered step1, active: ", self.active
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
        column_names = X.design_info.column_names
        print "entering case: ", self.groupVars
        active = self.active.copy()
        if not self.groupVars :
            if direction == "forward" :   
                winners = qr_based_solver.pickForwardVariable(X,
                                                            y,
                                                            self.active,
                                                            self.currentFit)
                assert(len(winners) == 1)
                print "winner: ", winners
                assert not active[winners[0]]
                active[winners[0]] = True
            else : #backward
                winners = qr_based_solver.pickBackwardVariable(X,
                                                            y,
                                                            self.active,
                                                            self.lower_active,
                                                            self.currentFit,
                                                            self.X_columns)
                assert len(winners) == 1
                assert active[winners[0]]
                active[winner[0]] = False
        else : #Grouped variables
            if direction == "backward" :
                print "entering pickBackwardVariable..."
                winners = qr_based_solver.delNextVariableGroupBruteForce(X,
                                                              y,
                                                              self.active,
                                                              self.lower_active,
                                                              self.penaltyFn)
                active[winners] = False
            else : 
                winners = qr_based_solver.pickNextVariableGroupBruteForce(X,
                                                y,
                                                active,
                                                X.design_info.term_slices,
                                                self.penaltyFn)
                assert np.all(np.logical_not(active[winners]))
                active[winners] = True
                 

        
        if np.all(active == self.active) :
            print "They are all the same!"
            raise StepwiseNoAction()
        beta, var, SSE, df, _ = qr_based_solver.solve(X[:,active],y,self.tolerance) 
        print "active: ", active, self.X_columns
        column_names = [ name for i,name in enumerate(self.X_columns) if active[i]  ]
        
        if self.penaltyFn.getValue(X.shape[0],df,SSE) <= \
                self.penaltyFn.getValue(X.shape[0],old_summary.degrees_freedom,old_summary.SSE):
            raise StepwiseNoAction()
        modelDesc = patsy.ModelDesc(self.y_columns,column_names)
        summary = computeSummary(beta,var,SSE,self.__SSTO,X.shape[0],df,
                                column_names,modelDesc)
        
        if self.trace : 
            print >>self.traceFile, "Step Count: ", self.stepCount
            summary.write(self.traceFile)
        self.active  = active
        self.currentFit = LinearModelFit(summary.beta,summary)
        self.stepCount += 1
    
    def step(self,direction="both") :
        count = 0
        if self.trace :
            print >>self.traceFile, "Step Count: ", self.stepCount
            self.currentFit.summary.write(self.traceFile)
        if direction == "forward" :
            try :
                while 1 :
                    self.step1("forward")
                    count += 1
            except StepwiseNoAction, e : pass
        elif direction == "backward" :
            try :
                while 1 :
                    self.step1("backward")
                    count += 1
            except StepwiseNoAction, e : pass
        elif direction == "both" :
            # a heuristic based on alternating between adding and deleting features
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
                count += 1
        else :
            raise StepwiseError("direction argument should be one of: 'forward', " +
                                "'backward', or 'both'")
        


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
    env = patsy.EvalEnvironment.capture()     
    upperScopeDesc = patsy.ModelDesc.from_formula(upperScope, env)
    startScopeDesc = None if startScope is None else ModelDesc.from_formula(startScope,env)
    lowerScopeDesc = None if lowerScope is None else ModelDesc.from_formula(lowerScope,env)
    if not lowerScope and patsy.Term([]) not in upperScopeDesc.rhs_termlist :
        raise StepwiseError("A lower scope of the model search must be specified when " + \
        "the upperScope does not contain an intercept")
    if not lowerScope : #build a formula with only an intercept
        lowerScopeDesc = patsy.ModelDesc(upperScopeDesc.lhs_termlist,[patsy.Term([])])
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

    
    fit = lm("sl ~ 1+sx+rk+yr+dg+yd+dg2",data,fitStrategy=QR)
    est,anova = fit.summary.textPretty()
    print est
    print anova
    fit.summary.write()
#    fit = lm("sl ~ 1+sx+rk+yr+dg+yd+yd2",data,fitStrategy=Adjust)
#
#    #fit = lm("sl ~  dg + yd +yd2", data)
#
#    fit.summary.write()
#
    #data = pandas.io.parsers.read_csv("salary2.txt")
    #stepper = stepwiseInit("sl ~ sx+rk+yr+dg+yd",data,trace=True,groupVars=True)
    #stepper.step(direction="both")
    #stepper = stepwiseInit("sl ~ sx+rk+yr+dg+yd",data,startScope="sl ~ sx+rk+yr+dg+yd",
    #trace=True,groupVars=False)
    #stepper.step(direction="backward")
    
    
    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    #data = sm.datasets.longley.load_pandas()
    #data = data.exog.join(data.endog)
    #print data
    #formula = "TOTEMP ~ 1 + GNP + UNEMP + GNPDEFL + POP + YEAR"
    #fit = lm(formula,data)
    #print fit.summary.write()
    #fit = ols(formula=formula, data=data).fit()
    #print fit.summary()
    #import pandas.rpy.common as com
    #import rpy2.robjects as robjects
    #r = robjects.r
    #r_dataframe = com.convert_to_r_dataframe(data)
    #lm = r.lm(formula,data=r_dataframe)
    #print r.summary(lm)
    
