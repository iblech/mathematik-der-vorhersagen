#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm trainiert ein neuronales Netzwerk mit einer Hidden-Schicht
# auf dem MNIST-Datensatz zur Erkennung handschriftlicher Ziffern.
#
# Es setzt nicht die L2-, sondern die Cross-Entropy-Kostenfunktion ein.
# Außerdem verwendet es L2-Regularisierung. Beide Verbesserungen sind
# auf http://neuralnetworksanddeeplearning.com/chap3.html erklärt.
#
# Mit 100 versteckten Neuronen erhält man etwa 97 % als Erfolgsrate.

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

    delta   = z - zbar
    gamma   = np.dot(W.T, delta) * sigma_(yhat)

    nabla_c = delta
    nabla_b = gamma

    nabla_W = np.dot(delta, y.T)
    nabla_V = np.dot(gamma, x.T)

    return nabla_V, nabla_W, nabla_b, nabla_c

# Gradient der CE-Kostenfunktion, berechnet als Summe der Gradienten
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
eta       = 0.5
kappa     = 0.1

# Zufällige Initialisierung von Gewichten und Biases.
V = np.random.normal(0, 1, (numHidden,numInput))
W = np.random.normal(0, 1, (numOutput,numHidden)) / np.sqrt(numHidden)
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
        V = V - eta * nabla_V / batchSize - eta * kappa * V / len(data)
        W = W - eta * nabla_W / batchSize - eta * kappa * W / len(data)
        b = b - eta * nabla_b / batchSize
        c = c - eta * nabla_c / batchSize

# Mit den Hyper-Parametern
#
#    versteckte Neuronen: 30
#    Batchgröße:          10
#    eta:                 0.5
#    kappa:               0.1
#
# sieht der Trainingsverlauf so aus:
#
# Epoch 00: 902 von 10000 Ziffern richtig erkannt.
# Epoch 01: 9158 von 10000 Ziffern richtig erkannt.
# Epoch 02: 9237 von 10000 Ziffern richtig erkannt.
# Epoch 03: 9339 von 10000 Ziffern richtig erkannt.
# Epoch 04: 9398 von 10000 Ziffern richtig erkannt.
# Epoch 05: 9438 von 10000 Ziffern richtig erkannt.
# Epoch 06: 9484 von 10000 Ziffern richtig erkannt.
# Epoch 07: 9472 von 10000 Ziffern richtig erkannt.
# Epoch 08: 9475 von 10000 Ziffern richtig erkannt.
# Epoch 09: 9495 von 10000 Ziffern richtig erkannt.
# Epoch 10: 9506 von 10000 Ziffern richtig erkannt.
# Epoch 11: 9501 von 10000 Ziffern richtig erkannt.
# Epoch 12: 9539 von 10000 Ziffern richtig erkannt.
# Epoch 13: 9523 von 10000 Ziffern richtig erkannt.
# Epoch 14: 9500 von 10000 Ziffern richtig erkannt.
# Epoch 15: 9554 von 10000 Ziffern richtig erkannt.
# Epoch 16: 9541 von 10000 Ziffern richtig erkannt.
# Epoch 17: 9527 von 10000 Ziffern richtig erkannt.
# Epoch 18: 9521 von 10000 Ziffern richtig erkannt.
# Epoch 19: 9506 von 10000 Ziffern richtig erkannt.
# Epoch 20: 9518 von 10000 Ziffern richtig erkannt.
# Epoch 21: 9536 von 10000 Ziffern richtig erkannt.
# Epoch 22: 9566 von 10000 Ziffern richtig erkannt.   *****
# Epoch 23: 9504 von 10000 Ziffern richtig erkannt.
# Epoch 24: 9547 von 10000 Ziffern richtig erkannt.
# Epoch 25: 9528 von 10000 Ziffern richtig erkannt.
# Epoch 26: 9548 von 10000 Ziffern richtig erkannt.
# Epoch 27: 9547 von 10000 Ziffern richtig erkannt.
# Epoch 28: 9543 von 10000 Ziffern richtig erkannt.
# Epoch 29: 9525 von 10000 Ziffern richtig erkannt.
#
# Mit 100 versteckten Neuronen erreichen wir:
# Epoch 00: 1135 von 10000 Ziffern richtig erkannt.
# Epoch 01: 9303 von 10000 Ziffern richtig erkannt.
# Epoch 02: 9331 von 10000 Ziffern richtig erkannt.
# Epoch 03: 9446 von 10000 Ziffern richtig erkannt.
# Epoch 04: 9520 von 10000 Ziffern richtig erkannt.
# Epoch 05: 9535 von 10000 Ziffern richtig erkannt.
# Epoch 06: 9582 von 10000 Ziffern richtig erkannt.
# Epoch 07: 9589 von 10000 Ziffern richtig erkannt.
# Epoch 08: 9599 von 10000 Ziffern richtig erkannt.
# Epoch 09: 9627 von 10000 Ziffern richtig erkannt.
# Epoch 10: 9638 von 10000 Ziffern richtig erkannt.
# Epoch 11: 9630 von 10000 Ziffern richtig erkannt.
# Epoch 12: 9644 von 10000 Ziffern richtig erkannt.
# Epoch 13: 9634 von 10000 Ziffern richtig erkannt.
# Epoch 14: 9651 von 10000 Ziffern richtig erkannt.
# Epoch 15: 9649 von 10000 Ziffern richtig erkannt.
# Epoch 16: 9665 von 10000 Ziffern richtig erkannt.
# Epoch 17: 9671 von 10000 Ziffern richtig erkannt.
# Epoch 18: 9647 von 10000 Ziffern richtig erkannt.
# Epoch 19: 9659 von 10000 Ziffern richtig erkannt.
# Epoch 20: 9666 von 10000 Ziffern richtig erkannt.
# Epoch 21: 9668 von 10000 Ziffern richtig erkannt.
# Epoch 22: 9658 von 10000 Ziffern richtig erkannt.
# Epoch 23: 9663 von 10000 Ziffern richtig erkannt.
# Epoch 24: 9673 von 10000 Ziffern richtig erkannt.
# Epoch 25: 9677 von 10000 Ziffern richtig erkannt.
# Epoch 26: 9658 von 10000 Ziffern richtig erkannt.
# Epoch 27: 9654 von 10000 Ziffern richtig erkannt.
# Epoch 28: 9681 von 10000 Ziffern richtig erkannt.
# Epoch 29: 9681 von 10000 Ziffern richtig erkannt.    ****
