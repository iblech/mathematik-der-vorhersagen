#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm erlaubt es, interaktiv ein neuronales Netz zur Ziffernerkunng
# zu testen. Mit der linken Maustaste malt man, mit der rechten wird das Bild
# zurückgesetzt.
#
# Achtung: Beim Zeichnen der Ziffern darauf aufpassen, dass man den Stil der
# MNIST-Ziffern möglichst gut einhält. Das heißt konkret: Die Ränder freilassen
# (das ist wirklich wichtig!) und Ziffern im amerikanischen Stil malen. Siehe:
# http://pavel.surmenok.com/wp-content/uploads/2014/07/mnistdigits.gif

import os
import sys
import numpy as np
import matplotlib
if sys.platform == "darwin":
    matplotlib.use('TkAgg')
    # Workaround um einen Bug.
    # Siehe https://github.com/mwaskom/seaborn/issues/231, danke an Robin!
import matplotlib.pyplot as plt
import pickle
import scipy.ndimage

# In das Verzeichnis wechseln, in dem sich dieses Programm befindet.
os.chdir(sys.path[0])

# Die Sigma-Funktion.
def sigma(t): return 1 / (1 + np.exp(-t))

# Feedforward durchs Netz. Gibt auch alle Zwischenergebnisse zurück.
def feedforward(x):
    global V,W,b,c
    yhat = np.dot(V,x) + b
    y    = sigma(yhat)
    zhat = np.dot(W,y) + c
    z    = sigma(zhat)
    return yhat, y, zhat, z

# Versuch, das gespeicherte Netz zu laden.
with open("net.p", "rb") as f:
    V,W,b,c = pickle.load(f, encoding="latin1")

# (28x28)-Array, das die Bilddaten enthält.
image = np.zeros((28,28))

fig = plt.figure()
ax  = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax3 = fig.add_subplot(133)

im  = ax.imshow(image, vmin=0, vmax=1)
ax.set_title("Draw a digit")
ax.axis("off")

bars = ax2.bar(range(10), np.zeros(10))
ax2.set_ylim(0, 1)
ax2.set_title("Output neuron activations")
ax2.set_xlabel("Digit")
ax2.set_ylabel("Activation")
ax2.set_xticks(range(10))

ax3.axis("off")
predicted_text = ax3.text(0.5, 0.5, "", fontsize=80, ha="center", va="center", color="red")
ax3.set_title("")

plt.tight_layout()
plt.draw()

# Gegeben zwei Punkte old und new, gibt insgesamt 100 Punkte auf der
# Verbindungsstrecke von old nach new zurück. Wird verwendet, weil die Maus
# bei schnellen Bewegungen oft einzelne Pixel auslässt.
def lininp(old, new):
    if old is None:
        yield new
        return

    for kappa in np.arange(0, 1, 0.01):
        yield (1-kappa)*old + kappa*new

# Entfernt eine führende Null, falls vorhanden. Zur übersichtlicheren
# Formatierung der Aktivität der Ausgabeneuronen.
def stripzero(str):
    if str[0] == '0':
        return str[1:]
    else:
        return str

oldpos = None
def onmouse(event):
    global image, ax, im, oldpos

    if event.button == 1:
        # Für alle Punkte auf der Verbindungsstrecke von der vorherigen
        # Position zur momentanen Position den entsprechenden Eintrag in der
        # Matrix auf Eins setzen.
        for p in lininp(oldpos, np.array([event.xdata, event.ydata])):
            image[int(round(p[1])),   int(round(p[0]))]   = 1
            image[int(round(p[1]-1)), int(round(p[0]))]   = 1
            image[int(round(p[1]+1)), int(round(p[0]))]   = 1
            image[int(round(p[1])),   int(round(p[0]-1))] = 1
            image[int(round(p[1])),   int(round(p[0]+1))] = 1
        oldpos = np.array([event.xdata, event.ydata])

        # Grafische Darstellung aktualisieren.
        im.set_data(image)
        ax.draw_artist(im)

        # Bild durchs neuronale Netz schicken.
        act = feedforward(image.reshape(784,1))[-1].flatten()
        print("%d" % np.argmax(act) + ": " + ", ".join([ stripzero("%1.4f" % a) for a in act ]))
        for rect, h in zip(bars, act):
            rect.set_height(h)
        ax2.draw_artist(ax2.patch)
        for bar in bars:
            ax2.draw_artist(bar)

        predicted_digit = np.argmax(act)
        predicted_text.set_text(str(predicted_digit))
        ax3.draw_artist(predicted_text)

        im.figure.canvas.blit(im.figure.bbox)
    else:
        oldpos = None

        # Bei Rechtsklick Bild zurücksetzen.
        if event.button == 3:
            image = np.zeros((28,28))
            im.set_data(image)
            ax.draw_artist(im)

            for rect in bars:
                rect.set_height(0)
            ax2.draw_artist(ax2.patch)

            predicted_text.set_text('')
            ax3.draw_artist(predicted_text)

            im.figure.canvas.blit(im.figure.bbox)

fig.canvas.mpl_connect('button_press_event',  onmouse)
fig.canvas.mpl_connect('motion_notify_event', lambda event: onmouse(event))

plt.show()
