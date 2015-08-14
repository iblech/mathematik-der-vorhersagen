#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm trainiert ein neuronales Netzwerk mit einer Hidden-Schicht
# auf dem MNIST-Datensatz zur Erkennung handschriftlicher Ziffern.
#
# Es geht über die im Kurs unmittelbar besprochenen und auch programmierten
# Ideen nicht hinaus. Der Klassifikationserfolgt liegt mit 30 Neuronen auf der
# Zwischenschicht bei etwa 95 %. Mit 100 Neuronen erreicht man etwa 97 %.

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import mnist_loader
import cPickle as pickle

# In das Verzeichnis wechseln, in dem sich dieses Programm befindet.
os.chdir(sys.path[0])

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

# Bestimmt, wie viele Ziffern in `testData` richtig erkannt werden.
def evaluate(V,W,b,c, testData):
    results = [(np.argmax(feedforward(V,W,b,c, x)[-1]), actualDigit) for (x, actualDigit) in testData]
    return sum(int(x == digit) for (x, digit) in results)

# Definition der Schichtgrößen.
numInput  = 784
numHidden = 30
numOutput = 10

# Definition der Hyperparameter.
numEpochs = 30
batchSize = 10
eta       = 3.0

# Zufällige Initialisierung von Gewichten und Biases.
V = np.random.normal(0, 1, (numHidden,numInput))
W = np.random.normal(0, 1, (numOutput,numHidden))
b = np.random.normal(0, 1, (numHidden,1))
c = np.random.normal(0, 1, (numOutput,1))

# Der MNIST-Datensatz.
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

for i in range(numEpochs):
    # Aktuellen Stand ausgeben.
    print("Epoch %02d: %d von %d Ziffern richtig erkannt." % (i, evaluate(V,W,b,c, testData), len(testData)))
    # Netzwerk speichern.
    with open("net.p", "wb") as f:
        pickle.dump((V,W,b,c), f)

    # Trainingsdaten durchmischen.
    data = list(trainingData)
    random.shuffle(data)

    # Nun für jede Batch jeweils einen Schritt in Richtung des umgekehrten
    # Gradienten gehen.
    for k in range(0, len(data), batchSize):
        batch = data[k:k+batchSize]

        nabla_V, nabla_W, nabla_b, nabla_c = gradient(V,W,b,c, batch)
        V = V - eta * nabla_V / batchSize
        W = W - eta * nabla_W / batchSize
        b = b - eta * nabla_b / batchSize
        c = c - eta * nabla_c / batchSize

# Mit den Hyper-Parametern
#
#    versteckte Neuronen: 30
#    Batchgröße:          10
#    eta:                 3.0
#
# sieht der Trainingsverlauf so aus:
#
# Epoch 00: 950 von 10000 Ziffern richtig erkannt.
# Epoch 01: 9137 von 10000 Ziffern richtig erkannt.
# Epoch 02: 9221 von 10000 Ziffern richtig erkannt.
# Epoch 03: 9324 von 10000 Ziffern richtig erkannt.
# Epoch 04: 9341 von 10000 Ziffern richtig erkannt.
# Epoch 05: 9343 von 10000 Ziffern richtig erkannt.
# Epoch 06: 9391 von 10000 Ziffern richtig erkannt.
# Epoch 07: 9362 von 10000 Ziffern richtig erkannt.
# Epoch 08: 9389 von 10000 Ziffern richtig erkannt.
# Epoch 09: 9434 von 10000 Ziffern richtig erkannt.
# Epoch 10: 9405 von 10000 Ziffern richtig erkannt.
# Epoch 11: 9441 von 10000 Ziffern richtig erkannt.
# Epoch 12: 9455 von 10000 Ziffern richtig erkannt.
# Epoch 13: 9451 von 10000 Ziffern richtig erkannt.
# Epoch 14: 9469 von 10000 Ziffern richtig erkannt.
# Epoch 15: 9427 von 10000 Ziffern richtig erkannt.
# Epoch 16: 9458 von 10000 Ziffern richtig erkannt.
# Epoch 17: 9467 von 10000 Ziffern richtig erkannt.
# Epoch 18: 9447 von 10000 Ziffern richtig erkannt.
# Epoch 19: 9403 von 10000 Ziffern richtig erkannt.
# Epoch 20: 9460 von 10000 Ziffern richtig erkannt.
# Epoch 21: 9450 von 10000 Ziffern richtig erkannt.
# Epoch 22: 9466 von 10000 Ziffern richtig erkannt.
# Epoch 23: 9461 von 10000 Ziffern richtig erkannt.
# Epoch 24: 9444 von 10000 Ziffern richtig erkannt.
# Epoch 25: 9456 von 10000 Ziffern richtig erkannt.
# Epoch 26: 9491 von 10000 Ziffern richtig erkannt.
# Epoch 27: 9458 von 10000 Ziffern richtig erkannt.
# Epoch 28: 9495 von 10000 Ziffern richtig erkannt.
# Epoch 29: 9499 von 10000 Ziffern richtig erkannt.
# Epoch 30: 9499 von 10000 Ziffern richtig erkannt.
# Epoch 31: 9493 von 10000 Ziffern richtig erkannt.
# Epoch 32: 9492 von 10000 Ziffern richtig erkannt.
# Epoch 33: 9490 von 10000 Ziffern richtig erkannt.
# Epoch 34: 9473 von 10000 Ziffern richtig erkannt.
# Epoch 35: 9501 von 10000 Ziffern richtig erkannt.
# Epoch 36: 9506 von 10000 Ziffern richtig erkannt.
# Epoch 37: 9484 von 10000 Ziffern richtig erkannt.
# Epoch 38: 9501 von 10000 Ziffern richtig erkannt.
# Epoch 39: 9475 von 10000 Ziffern richtig erkannt.
# Epoch 40: 9474 von 10000 Ziffern richtig erkannt.
# Epoch 41: 9455 von 10000 Ziffern richtig erkannt.
# Epoch 42: 9488 von 10000 Ziffern richtig erkannt.
# Epoch 43: 9490 von 10000 Ziffern richtig erkannt.
# Epoch 44: 9491 von 10000 Ziffern richtig erkannt.
# Epoch 45: 9472 von 10000 Ziffern richtig erkannt.
# Epoch 46: 9479 von 10000 Ziffern richtig erkannt.
# Epoch 47: 9507 von 10000 Ziffern richtig erkannt.   ****
# Epoch 48: 9496 von 10000 Ziffern richtig erkannt.
# Epoch 49: 9480 von 10000 Ziffern richtig erkannt.
# Epoch 50: 9492 von 10000 Ziffern richtig erkannt.
# Epoch 51: 9462 von 10000 Ziffern richtig erkannt.
# Epoch 52: 9484 von 10000 Ziffern richtig erkannt.
# Epoch 53: 9490 von 10000 Ziffern richtig erkannt.
