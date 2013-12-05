============
markfit
============

This module contains several pieces of functionality related to
fitting linear models.  For example, it contains a linear regression
package which employs formulas from the patsy package.

Requirements

numpy >= 1.7
scipy >= 0.12.0
pandas >= 0.12.0
patsy >= 0.2.1 


import pandas
from markfit import leastsq

data  = pandas.io.parsers.read_csv("test_data.csv")
model = leastsq.fit("y~x",data)
model.summary.write()

#Here is an example running stepwise regression:

stepper = leastsq.stepwiseInit("y ~ sx+x+yr+dg+yd",data,trace=True,groupVars=True)
stepper.step(direction="both")

The package also includes some numerical linear algebra code useful
for fitting linear regression and constrained optimization problems.
The cleastsq directory contains much of this work and includes
code for householder computations, QR decomposition, LQ
decomposition.  One of the goals of releasing the package is to collaborate
with others as we clean up the numerical procedures and use them as building
blocks for features not currently found in Python implementations of
linear regression.

