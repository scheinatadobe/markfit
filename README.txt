============
pyleastsq
============

This module contains several pieces of functionality related to
fitting linear models.  For example, it contains a linear regression
package which employs formulas from the patsy package 

import patsy
import pandas.io
from markfit import leastsq
from sys import stdout

data  = pandas.io.parsers.read_csv("test_data.csv")
model = leastsq.base.lm("y~x",data) 
model.summary.write(stdout)

The package also includes some numerical linear algebra code useful
for fitting linear regression and constrained optimization problems.
The constrained_ls directory contains much of this work and includes
code for householder computations, QR decomposition, LQ
decomposition.  One of the goals of releasing the package is to clean
up and optimize these numerical procedures and then use them as
building blocks for stepwise regression and constrained optimization
implementations.


