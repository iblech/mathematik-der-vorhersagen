#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm trainiert ein neuronales Netzwerk der Form
#
#     *-----*-----*
#
# so, dass es die Zuordnung
#
#     0 |--> 0
#     5 |--> 0
#    20 |--> 1
#    25 |--> 1
#
# realisiert. Zusätzlich wird L2-Regularisierung angewendet; wenn man diese
# deaktivieren möchte, muss man unten `kappa` auf 0 setzen.
#
# Die Schrittweite für den Gradientenabstieg ist nicht fix, sondern wird
# adaptiv angepasst.
#
# Dieses Programm plottet die Feedforward-Funktion nach jedem Schritt des
# Gradientenabstiegs in einem Unterordner "images". Die einzelnen Plots kann
# man unter Linux mit dem Befehl
#
#     ffmpeg -i images/abschneidefunktion-%4d.png abschneidefunktion.mp4
#
# zu einer Video-Datei kombinieren (ffmpeg ist im Paket libav-tools enthalten).

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    os.mkdir("images")
except:
    pass

def sigma(t):  return 1/(1 + np.exp(-t))
def sigma_(t): return sigma(t)*(1-sigma(t))

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
            eta = eta * 1.2  # der Schritt war gut; Schrittweite etwas erhöhen
            yield x
        else:
            eta = eta * 0.5  # der Schritt wäre schlecht; wir gehen ihn nicht
                             # und verringern die Schrittweite

# Kostenfunktion, inklusive Strafterm zur L2-Regularisierung.
def ex_f(x):
    w, c = x[0], x[1]
    kappa = 0.1
    return sigma(5*w+c)**2 + (sigma(25*w+c)-1)**2 + sigma(0*w+c)**2 + (sigma(20*w+c)-1)**2 + kappa*w**2 + kappa*c**2

# Gradient der Kostenfunktion.
def ex_fderiv(x):
    w, c = x[0], x[1]
    kappa = 0.1
    return np.array([2*sigma(5*w+c)*sigma_(5*w+c)*(5) + 2*(sigma(25*w+c)-1)*sigma_(25*w+c)*25 + 2*sigma(0*w+c)*sigma_(0*w+c)*0 + 2*(sigma(20*w+c)-1)*sigma_(20*w+c)*20 + 2*kappa*w, 2*sigma(5*w+c)*sigma_(5*w+c) + 2*(sigma(25*w+c)-1)*sigma_(25*w+c) + 2*sigma(0*w+c)*sigma_(0*w+c) + 2*sigma(20*w+c)*sigma_(20*w+c) + 2*kappa*c])

xs  = np.arange(-15, 28, 0.1)
yy  = np.zeros(len(xs))
i = 0
fig = plt.figure()
for x in descent(ex_f, ex_fderiv, np.array([0,0]), 0.001):
    print(x, ex_f(x))
    i = i + 1
    if i > 1000: break

    w, c = x
    plt.cla()
    plt.axis([-10, 25, 0, 1])
    plt.plot(xs, sigma(xs*w+c))
    fig.savefig('images/abschneidefunktion-%04d.png' % i, dpi=fig.dpi)
