#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm trainiert ein neuronales Netzwerk mit einer Hidden-Schicht
# auf dem MNIST-Datensatz zur Erkennung handschriftlicher Ziffern.
#
# Es geht über die im Kurs unmittelbar besprochenen und auch programmierten
# Ideen nur in einem kleinen Detail hinaus: Die Gewichte in V und W werden
# mit einem bestimmten Faktor, der von der Anzahl der Verbindungen zwischen den
# Neuronen abhängt, skaliert. Damit soll das Training beschleunigt werden.
#
# Der Klassifikationserfolg liegt bei 30 versteckten Neuronen etwa bei 95 %.
# Mit 100 versteckten Neuronen erhält man sogar 97,4 %.

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import mnist_loader
import pickle

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
V = np.random.normal(0, 1, (numHidden,numInput))  / np.sqrt(numInput)
W = np.random.normal(0, 1, (numOutput,numHidden)) / np.sqrt(numHidden)
b = np.random.normal(0, 1, (numHidden,1))
c = np.random.normal(0, 1, (numOutput,1))

# Der MNIST-Datensatz.
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

for i in range(numEpochs):
    # Aktuellen Stand ausgeben.
    print("Epoch %02d: %d von %d Ziffern richtig erkannt." % (i, evaluate(V,W,b,c, testData), len(testData)), file=sys.stderr)
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

        print(*[a[0] for a in feedforward(V,W,b,c, validationData[0][0])[-1]])

# Mit den Hyper-Parametern
#
#    versteckte Neuronen: 30
#    Batchgröße:          10
#    eta:                 3.0
#
# sieht der Trainingsverlauf so aus:
#
# Epoch 00: 863 von 10000 Ziffern richtig erkannt.
# Epoch 01: 9076 von 10000 Ziffern richtig erkannt.
# Epoch 02: 9265 von 10000 Ziffern richtig erkannt.
# Epoch 03: 9350 von 10000 Ziffern richtig erkannt.
# Epoch 04: 9370 von 10000 Ziffern richtig erkannt.
# Epoch 05: 9400 von 10000 Ziffern richtig erkannt.
# Epoch 06: 9421 von 10000 Ziffern richtig erkannt.
# Epoch 07: 9397 von 10000 Ziffern richtig erkannt.
# Epoch 08: 9463 von 10000 Ziffern richtig erkannt.
# Epoch 09: 9495 von 10000 Ziffern richtig erkannt.
# Epoch 10: 9513 von 10000 Ziffern richtig erkannt.
# Epoch 11: 9467 von 10000 Ziffern richtig erkannt.
# Epoch 12: 9427 von 10000 Ziffern richtig erkannt.
# Epoch 13: 9482 von 10000 Ziffern richtig erkannt.
# Epoch 14: 9456 von 10000 Ziffern richtig erkannt.
# Epoch 15: 9483 von 10000 Ziffern richtig erkannt.
# Epoch 16: 9504 von 10000 Ziffern richtig erkannt.
# Epoch 17: 9516 von 10000 Ziffern richtig erkannt.
# Epoch 18: 9515 von 10000 Ziffern richtig erkannt.
# Epoch 19: 9511 von 10000 Ziffern richtig erkannt.
# Epoch 20: 9529 von 10000 Ziffern richtig erkannt.
# Epoch 21: 9531 von 10000 Ziffern richtig erkannt.    ****
# Epoch 22: 9514 von 10000 Ziffern richtig erkannt.
# Epoch 23: 9483 von 10000 Ziffern richtig erkannt.
# Epoch 24: 9512 von 10000 Ziffern richtig erkannt.
# Epoch 25: 9518 von 10000 Ziffern richtig erkannt.
# Epoch 26: 9508 von 10000 Ziffern richtig erkannt.
# Epoch 27: 9495 von 10000 Ziffern richtig erkannt.
# Epoch 28: 9513 von 10000 Ziffern richtig erkannt.
# Epoch 29: 9515 von 10000 Ziffern richtig erkannt.
