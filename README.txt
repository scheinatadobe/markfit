============
pyleastsq
============

This module contains several pieces of functionality related to
fitting linear models.  For example, it contains a linear regression
package which employs formulas from the patsy package 

from pyleastsq import lm
data = pandas.io.parsers.read_csv("salary2.txt")
fit = lm("sl ~ 1+sx+rk+yr+dg+yd+dg2",data,fitStrategy=QR)
fit.summary.write()

The package also includes some numerical linear algebra code useful
for fitting linear regression and constrained optimization problems.
The constrained_ls directory contains much of this work and includes
code for householder computations, QR decomposition, LQ
decomposition.  One of the goals of releasing the package is to clean
up and optimize these numerical procedures and then use them as
building blocks for stepwise regression and constrained optimization
implementations.


