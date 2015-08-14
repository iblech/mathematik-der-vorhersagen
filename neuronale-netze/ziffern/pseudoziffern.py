#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm verwendet ein bereits trainiertes Netz zur Ziffernerkennung,
# gespeichert in der Datei "net.p", um ein Ziffer des MNIST-Datensatzes
# schrittweise so zu ändern, dass es als eine bestimmte andere Ziffer erkannt wird.
#
# Die Einzelbilder werden in einem Ordner "images" abgelegt.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import mnist_loader
import cPickle as pickle

# In das Verzeichnis wechseln, in dem sich dieses Programm befindet.
os.chdir(sys.path[0])

# Die Sigma-Funktion und ihre Ableitung..
def sigma(t):  return 1 / (1 + np.exp(-t))
def sigma_(t): return sigma(t) * (1 - sigma(t))

# Feedforward durchs Netz. Gibt auch alle Zwischenergebnisse zurück.
def feedforward(V,W,b,c, x):
    yhat = np.dot(V,x) + b
    y    = sigma(yhat)
    zhat = np.dot(W,y) + c
    z    = sigma(zhat)
    return yhat, y, zhat, z

# Gespeichertes Netzwerk laden.
with open("net.p", "rb") as f:
    V,W,b,c = pickle.load(f)

# Der MNIST-Datensatz.
trainingData, validationData, testData = mnist_loader.load_data_wrapper()

# Ausgangsbild.
x = trainingData[0][0]  # oder: np.random.uniform(0, 1, (784,1))
plt.imshow(np.reshape(x, (28, 28)))
plt.show()

target = np.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]])
eta    = 0.001

try:
    os.mkdir("images")
except:
    pass

# Was passiert hier? Es handelt sich um ein Gradientenabstiegsverfahren mit
# adaptischer Schrittweitenkontrolle. Die Funktion, die minimiert wird, ist
#
#     ||z - target||**2  +  0.0001 * ||x||.
#
# Der erste Term wird klein, wenn die Ausgabe des neuronalen Netzes nahe an der
# gewünschten Ausgabe liegt. Der zweite Term wird klein, wenn die Zahlen in x
# nicht zu groß sind. Das ist die "L2-Regularisierung". Man kann auf sie auch
# verzichten, dann erhält man einen leicht anderen Verlauf.

for i in range(3000):
    yhat, y, zhat, z = feedforward(V,W,b,c, x)

    # Aktuellen Wert von x als 28x28-Pixel-Bild ausgeben.
    if i % 1 == 0:
        plt.imshow(np.reshape(x, (28, 28)), cmap = pyl.cm.gray)
        plt.title(", ".join([ "%2.1e" % a for a in z ]))
        plt.savefig("images/pseudoziffer-%07d.png" % i)

    print(i, eta, z[0:4,:].T)

    newx = x - eta * (np.dot(z.T - target.T, np.dot(np.diag(sigma_(zhat)[:,0]), np.dot(W, np.dot(np.diag(sigma_(yhat)[:,0]), V)))).T + 0.0001 * x)

    # Hat sich der Abstand zur gewünschten Ausgabe verringert?
    if np.linalg.norm(feedforward(V,W,b,c, newx)[-1] - target)**2 < np.linalg.norm(z - target)**2:
        # Ja! Das neue x akzeptieren und eta etwas erhöhen.
        x   = newx
        eta = eta * 1.01
    else:
        # Leider nein! Die Schrittweite eta verringern.
        eta = eta * 0.5
