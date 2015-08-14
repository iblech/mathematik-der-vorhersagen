#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

def descent(f, fderiv, x0, eta0):
    x   = x0
    eta = eta0

    yield x

    while True:
        x_ = x - eta * fderiv(x)
        if f(x_) == f(x):
            x   = x_
            eta = eta * 0.5
            yield x
        elif f(x_) < f(x):
            x   = x_
            eta = eta * 1.2
            yield x
        else:
            eta = eta * 0.5

def ex_f(x):      return x**2 - 4*x - 7 + x*x*x - 100*math.sin(x)
def ex_fderiv(x): return 2*x  - 4       + 3*x*x - 100*math.cos(x)

i = 0
for x in descent(ex_f, ex_fderiv, 10, 0.001):
    print("%f\t%f" % (x, ex_f(x)))
    i = i + 1
    if i > 100: break

# In Gnuplot:
# plot x**2.-4.*x-7.+x**3.-100*sin(x) w l, "<python gradient-descent.py" w lp
