""" Collectin of penalty functions for use in stepwise regression. 
    We implement these as classes, so they can keep track of their human readable name for 
    report generation.
"""
import math

#def AIC(n,df,SSE) :
#    return -df - n * math.log(SSE/n)



class AIC :
    """ Akaike's Information Criterion """
    def __init__(self) : 
        self.__name = "AIC"
    def getValue(self,n,df,SSE) :
        return -df - n * math.log(SSE/n)
    def getName(self) : return self.__name

class AICc :
    """ AIC with a correction for finite sample size """
    def __init__(self) : 
        self.__name = "AICc"
    def getValue(self,n,df,SSE) :
        assert(df < n - 1)
        return -df - n * math.log(SSE/n) + 2*df*(df+1)/float(n - df - 1)
    def getName(self) : return self.__name



