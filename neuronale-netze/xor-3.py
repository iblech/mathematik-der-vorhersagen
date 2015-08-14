#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm trainiert ein neuronales Netzwerk der folgenden Struktur
# so, dass es dir XOR-Funktion berechnet.
#
#     *--+-----*
#         \   / \
#          \ /   \
#           /     *--
#          / \   /
#         /   \ /
#     *--+-----*
#
# Die Ausgabe des Programms sollte sich dem Ziel, [0,1,1,0], immer mehr
# annähern. Möglicherweise hat man Pech bei der zufälligen Wahl des
# Startpunkts. In diesem Fall kann man das Programm einfach erneut aufrufen und
# auf mehr Glück hoffen oder an dem Schrittweitenparameter eta drehen.

import numpy as np

# Die Sigma-Funktion und ihre Ableitung.
def sigma(t):  return 1 / (1 + np.exp(-t))
def sigma_(t): return sigma(t) * (1 - sigma(t))

# Feedforward durchs Netz. Gibt auch alle Zwischenergebnisse zurück.
def feedforward(V,W,b,c, x):
    yhat = np.dot(V,x) + b
    y    = sigma(yhat)
    zhat = np.dot(W,y) + c
    z    = sigma(zhat)
    return yhat, y, zhat, z

# Backpropagation für ein einzelnes (x,zbar)-Paar und die gewöhnliche
# L2-Kostenfunktion.
def backprop(V,W,b,c, x, zbar):
    yhat, y, zhat, z = feedforward(V,W,b,c, x)

    nabla_z = 2 * (z - zbar)

    delta   = nabla_z * sigma_(zhat)
    gamma   = np.dot(W.T, delta) * sigma_(yhat)

    nabla_c = delta
    nabla_b = gamma

    nabla_W = np.dot(delta, y.T)
    nabla_V = np.dot(gamma, x.T)

    return nabla_V, nabla_W, nabla_b, nabla_c

# Gradient der (L2-)Kostenfunktion, berechnet als Summe der Gradienten
# der einzelnen Summanden (ein Summand je Trainingsdatensatz).
def gradient(V,W,b,c, trainingData):
    nabla_V = np.zeros(V.shape)
    nabla_W = np.zeros(W.shape)
    nabla_b = np.zeros(b.shape)
    nabla_c = np.zeros(c.shape)

    for x, zbar in trainingData:
        part_nabla_V, part_nabla_W, part_nabla_b, part_nabla_c = backprop(V,W,b,c, x, zbar)
        nabla_V = nabla_V + part_nabla_V
        nabla_W = nabla_W + part_nabla_W
        nabla_b = nabla_b + part_nabla_b
        nabla_c = nabla_c + part_nabla_c

    return nabla_V, nabla_W, nabla_b, nabla_c

# Definition der Schichtgrößen.
numInput  = 2
numHidden = 2
numOutput = 1

# Zufällige Initialisierung von Gewichten und Biases.
V = np.random.normal(0, 1, (numHidden,numInput))
W = np.random.normal(0, 1, (numOutput,numHidden))
b = np.random.normal(0, 1, (numHidden,1))
c = np.random.normal(0, 1, (numOutput,1))

# Die Trainingsdaten für XOR.
trainingData = [(np.array([[0],[0]]), np.array([[0]])), (np.array([[0],[1]]), np.array([[1]])), (np.array([[1],[0]]), np.array([[1]])), (np.array([[1],[1]]), np.array([[0]]))]

# Die Schrittweite für den Gradientenabstieg.
eta = 0.5

for i in range(5000):
    # Aktuellen Stand ausgeben.
    print([feedforward(V,W,b,c, t[0])[3] for t in trainingData])

    # Gradientenabstieg durchführen.
    nabla_V, nabla_W, nabla_b, nabla_c = gradient(V,W,b,c, trainingData)
    V = V - eta * nabla_V
    W = W - eta * nabla_W
    b = b - eta * nabla_b
    c = c - eta * nabla_c
