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
from sys import stdout, stderr

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
    def __init__(self,n,p,beta,SSTO,SSE,MSE,SSR,MSR,f,pVal,RSq,paramT,paramPVal,S,column_names,
                formula) :
        self.n = n
        self.p = p
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
        self.formula   = formula

    def write(self,stdout=stdout,pad=4) :
        
        lines = []
        #X = self.X
        #if self.model.fit_intercept :
        #    lines.append("(Intercept)" + " "*pad + "intvalue") #TODO FIX
        #assert(len(X.design_info.column_names) == self.model.coef_.shape[1]) 
        print >>stdout, "Formula:", self.formula
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
        lines.append("SSR          %8.4g %12d %8.4g"%(self.SSR,self.p-1,self.MSR))
        lines.append("SSE          %8.4g %12d %8.4g"%(self.SSE,self.n-self.p,self.MSE))
        lines.append("")
        lines.append("F-statistic: %d"%int(self.f) + \
        " on %d and %d degrees of freedom"%(self.p-1,self.n - self.p) + \
                     ", p-value: %s"%pValueText)
        lines.append("Multiple R Squared: %4.3f"%self.RSq)
        lines.append("")
        print >>stdout, "\n".join(lines)     

def computeSummary(beta,betaSigmaSq,SSE,SSTO,n,p,column_names,formula) :
    """
    Summaryize the linear regression model
    #y    - the dependent variable
    #X    - the design matrix
    A    - the augmented matrix after solving
    n    - number of observations
    p    - number of predictors
    SSTO - the total sum of squares (can be computed without running a regression)
    returns a summary object
    
    The design of this summary is to not use X or y directly.  In this manner it might be used
    in a map-reduce context where N is too large to fit X or y in memory.

    """
    
    assert n>p
    assert p>1
    MSE        =   SSE / (n - p)
    SSR        =   SSTO - SSE #fundamental property of linear regression     
    MSR        =   SSR/(p - 1)
    f_stat     =   MSR/MSE #TODO: sensible checking
    pVal       =   1 - stats.f.cdf(f_stat,p-1,n-p)
    RSq        =   SSR / SSTO
    #On to estimation of parameter-specific values
    #for now hard code single response
    S  = np.sqrt(betaSigmaSq * MSE) #estimate parameter sigma
    paramT     = beta / S
    paramT[S == 0] = 0
    paramPVal  = 2 * (1 - stats.t.cdf(np.abs(paramT), n - p))
    return LinearModelSummary(n,p,beta,SSTO,SSE,MSE,SSR,MSR,f_stat,pVal,RSq,paramT,paramPVal,S,
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
        beta,betaSigmaSq,SSE = qr_based_solver.solve(X,y,tolerance=tolerance)
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
    summary = computeSummary(beta,betaSigmaSq,SSE,SSTO,n,p,X.design_info.column_names,formula)
    #print summary.beta
    return LinearModelFit(summary.beta,summary)

if __name__ == "__main__" :
    data = pandas.io.parsers.read_csv("salary2.txt")
    fit = lm("sl ~ 1+sx+rk+yr+dg+yd+yd2",data,fitStrategy=QR)
    fit.summary.write()
    fit = lm("sl ~ 1+sx+rk+yr+dg+yd+yd2",data,fitStrategy=Adjust)

    #fit = lm("sl ~  dg + yd +yd2", data)

    fit.summary.write()


