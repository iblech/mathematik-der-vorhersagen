#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm gibt die ersten neun Ziffern des MNIST-Datensatzes aus.
#
# Wenn ein gespeichertes neuronales Netz in der Datei "net.p" existiert,
# versucht das Programm außerdem, diese zu laden. Gelingt das, werden neun
# Klassifikationserfolge und neun Misserfolge angezeigt.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import mnist_loader
import pickle
import random

# In das Verzeichnis wechseln, in dem sich dieses Programm befindet.
os.chdir(sys.path[0])

# Die Sigma-Funktion.
def sigma(t): return 1 / (1 + np.exp(-t))

# Feedforward durchs Netz. Gibt auch alle Zwischenergebnisse zurück.
def feedforward(V,W,b,c, x):
    yhat = np.dot(V,x) + b
    y    = sigma(yhat)
    zhat = np.dot(W,y) + c
    z    = sigma(zhat)
    return yhat, y, zhat, z

def evaluate(V,W,b,c, testData):
    results = [(np.argmax(feedforward(V,W,b,c, x)[-1]), actualDigit) for (x, actualDigit) in testData]
    return sum(int(x == digit) for (x, digit) in results)

# Der MNIST-Datensatz.
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

a = plt.figure(1)
a.suptitle("Die ersten neun Ziffern des MNIST-Datensatzes", fontsize=14)
for k in range(9):
    ax = plt.subplot(3, 3, k+1)
    plt.imshow(trainingData[k][0].reshape(28,28), cmap = pyl.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Ziffer Nr. %d" % k)
plt.show()

print("Versuche, in net.p ein neuronales Netz zu laden...")
V,W,b,c = pickle.load(open("net.p", "rb"), encoding="latin1")
print("Erfolg! Berechne Erkennungsrate...")

print("Erkennungsrate: %d/%d" % (evaluate(V,W,b,c,testData), len(testData)))

correctResults = []
wrongResults   = []
random.shuffle(testData)
for x, actualDigit in testData:
    detectedDigit = np.argmax(feedforward(V,W,b,c, x)[-1])
    if detectedDigit == actualDigit:
        correctResults.append((x,actualDigit,detectedDigit))
    else:
        wrongResults  .append((x,actualDigit,detectedDigit))
    if len(correctResults) >= 16 and len(wrongResults) >= 16:
        break

a = plt.figure(1)
a.suptitle("Einige richtig erkannte Ziffern", fontsize=14)
for k in range(16):
    x, actualDigit, detectedDigit = correctResults[k]
    ax = plt.subplot(4, 4, k+1)
    plt.imshow(x.reshape(28,28), cmap = pyl.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Ist %d, erkannt als %d" % (actualDigit, detectedDigit))

a = plt.figure(2)
a.suptitle("Einige falsch erkannte Ziffern", fontsize=14)
for k in range(16):
    x, actualDigit, detectedDigit = wrongResults[k]
    ax = plt.subplot(4, 4, k+1)
    plt.imshow(x.reshape(28,28), cmap = pyl.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Ist %d, erkannt als %d" % (actualDigit, detectedDigit))
plt.show()

a = plt.figure(1)
a.suptitle("Gewichte der ersten 30 versteckten Neuronen", fontsize=14)
for k in range(30):
    ax = plt.subplot(6, 5, k+1)
    plt.imshow(V[k,:].reshape(28,28), cmap = pyl.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
plt.show()
