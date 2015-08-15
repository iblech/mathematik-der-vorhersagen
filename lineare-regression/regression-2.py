#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm zeichnet zu einer bestimmten Punktwolke die lineare,
# quadratische, kubische und quartische Ausgleichskurve.
#
# Die Koeffizienten der Ausgleichskurve werden über die Normalengleichung
# bestimmt.

import numpy as np
import matplotlib.pyplot as plt

# Diese Funktion legt durch gegebene Datenpunkte eine Ausgleichskurve.
#
# `x`:  Vektor der x-Stellen der Datenpunkte
# `y`:  Vektor der zugehörigen y-Werte
# `m`:  Grad der Ausgleichskurve (zum Beispiel führt m = 2 zu einer Parabel)
# `x_`: Vektor derjenigen x-Stellen, an denen die Ausgleichskurve geplottet werden soll
#
# Die Funktion gibt einen Vektor der zu den x-Stellen in `x_` gehörigen
# y-Werten der Ausgleichskurve zurück.
def fit(x, y, m, x_):
    n = len(x)

    # Matrix aufstellen.
    X = np.zeros((n, m+1))
    for i in range(n):
        for j in range(m+1):
            X[i,j] = x[i]**j

    # Das lineare Gleichungssystem (X^T X) a = X^T y nach a auflösen.
    a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    # Damit ist der Funktionsterm der Ausgleichskurve bekannt:
    # f(t) = a[0] + a[1]*t + a[2]*t*t + a[3]*t*t*t + ... + a[m]*t**m

    # Nun die y-Werte der Ausgleichskurve an den Stellen in `x_` berechnen.
    y_ = np.zeros(len(x_))
    for k in range(len(x_)):
        y_[k] = 0
        for l in range(m+1):
            y_[k] = y_[k] + a[l]*x_[k]**l
    return y_

# Beispieldatenpunkte.
x = np.arange(-2.1, 2.1, 0.05)
y = x**4 - 2*x**2 + x + np.random.normal(0,0.5, len(x))
plt.plot(x, y, "ro")

# Plot der Ausgleichskurven
x_ = np.arange(-2.1, 2.1, 0.1)
plt.plot(x_, fit(x, y, 1, x_))
plt.plot(x_, fit(x, y, 2, x_))
plt.plot(x_, fit(x, y, 3, x_))
plt.plot(x_, fit(x, y, 4, x_))

plt.show()
