#In this module, we write code to handle the more high level interfaces to linear modelling.
#Users provide high level details such as a data frame and formula, and the code
#here dispatches to the appropriate solvers
import numpy as np
import patsy
import pandas
#from sweep_based_solver import solve

from adjust_based_solver import solve 

import scipy.stats as stats
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

def computeSummary(A,n,p,SSTO,column_names,formula) :
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
    
    XXn1       =   A[:p,:p]
    XXn1        =   A[:p,p+1:]
    #print "diag: ", np.diag(XXn1)
    beta       =   A[:p,p]
    assert n>p
    assert p>1
    #yHat       =   model.predict(X)
    #if len(yHat.shape) > 1 and yHat.shape[1] == 1 : yHat = yHat.reshape(yHat.shape[0])
    #residue    =   y - yHat
    #ones       =   np.ones([y.shape[0],y.shape[0]])
    #SSE         =   np.dot(residue,residue)
    SSE        = A[p,p]
    #print "SSE: ", SSE
    MSE        =   SSE / (n - p)
    #print "MSE: ", MSE, np.diag(XXn1*MSE)
    #yBar       =   np.mean(y)
    #residue    =   yBar - yHat
    #SSR        =   np.dot(residue,residue)
    SSR        =   SSTO - SSE #fundamental property of linear regression     
    MSR        =   SSR/(p-1)
    f_stat     =   MSR/MSE #TODO: sensible checking
    pVal       =   1 - stats.f.cdf(f_stat,p-1,n-p)
    RSq        =   SSR / SSTO
    #On to estimation of parameter-specific values
    #for now hard code single response
    #print XXn1.shape, np.diag(XXn1)
    S  = np.sqrt(np.diag(XXn1 * MSE)) #estimate parameter sigma
    
    #print "S:", S
    paramT     = beta / S
    paramT[S == 0] = 0
    #self.SE    = self.S[0] / math.sqrt(self.n-self.p)
    paramPVal  = 2 * (1 - stats.t.cdf(np.abs(paramT), n - p))
    #print "shape for S: ", self.S.shape
    #print "shape for t : ", self.t.shape
    return LinearModelSummary(n,p,beta,SSTO,SSE,MSE,SSR,MSR,f_stat,pVal,RSq,paramT,paramPVal,S,
                                column_names,formula)

def lm(formula,dataFrame,fitStrategy=Adjust) :

    """ 
    formula   - a patsy formula
    dataFrame - a pandas data frame
    computeSummary - boolean, indicates whether we should compute ANOVA and variable info

    We make summary computation optional because it adds overhead
    """
    y,X = patsy.dmatrices(formula,dataFrame)    
    #print X.design_info.term_names    
    #print y.shape
    assert y.shape[1] == 1, "Multiple dependent variables are not yet supported."
    
    if fitStrategy == QR :
	raise "Not implemented yet!"	
    elif fitStrategy == "Sweep" :
	raise "Not implemented yet!"
	
    y = y.flatten()
    n   = y.shape[0]
    p   = X.shape[1]
    
    XX  = np.dot(X.T,X)
    Xy  = np.array([np.dot(X.T,y)])
    #print "p: ", p
    #print XX.shape, Xy.shape
    #build augmented matrix "A" to perform elimination on
    A = np.append(XX,Xy.T,axis=1)
    #print "Ashape: ", A.shape
    A = np.append(A,np.identity(XX.shape[0]),axis=1)

    #print "A first row shape: ", A.shape
    yy = np.array([[np.dot(y,y)]])
    #print Xy.shape, yy.shape
    bottom = np.append(Xy,yy.T,1)
    bottom = np.append(bottom,np.zeros([1,p]),1)
    #print "A.shape: ", A.shape, "bottom.shape: ", bottom.shape
    A = np.append(A,bottom,0)
    #print "A diag: ", np.diag(A)
    solve(A) 
    #print "solved:", A.shape
    #print A[:p,:p]
    #print A[:p,p+1:]
     
    #The problem is solved.  On to computing ANOVA table and other diagnostics! 
    residWithMean = y - np.mean(y)
    SSTO = np.dot(residWithMean,residWithMean)
    summary = computeSummary(A,n,p,SSTO,X.design_info.column_names,formula)
    #print summary.beta
    return LinearModelFit(summary.beta,summary)

if __name__ == "__main__" :
    data = pandas.io.parsers.read_csv("salary2.txt")
    fit = lm("sl ~ 1+sx+rk+yr+dg+yd",data)
    #fit = lm("sl ~  dg + yd +yd2", data)

    fit.summary.write()


