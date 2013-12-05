""" 
Collection of penalty functions for use in stepwise regression. 
We implement these as classes, so they can keep track of their human readable name for 
report generation.
"""
import math

class AIC :
    """ Akaike's Information Criterion """
    def __init__(self) : 
        self.__name = "AIC"
    def getValue(self,n,k,SSE) :
        return -k - n * math.log(SSE/n)
    def getName(self) : 
        return self.__name

class AICc :
    """ AIC with a correction for finite sample size """
    def __init__(self) : 
        self.__name = "AICc"
    def getValue(self,n ,k,SSE) :
        assert(k < n - 1)
        return -k - n * math.log(SSE/n) + 2*k*(k+1)/float(n - k - 1)
    def getName(self) : 
        return self.__name

class BIC :
    """ Bayesian Information Criterion """
    def __init__(self):
        self.__name = "BIC"
    def getValue(self,n,df,SSE):
        MSE = SSE/max(1.0,float(n))
        return n * math.log(MSE) + df * log(n)
    def getName(self): 
        return self.__name
    
class RIC :
    def __init__(self):
        self.__name = "BIC"
    def getValue(self,n,df,SSE):
        MSE = SSE/max(1.0,float(n))
        return SSE + SSE/max(1.0,MSE) * 2 * math.log(k)
    def getName(self) :
        return self.__name
    
