#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm erzeugt aus einem gegebenen Korpus an Text (zum Beispiel den
# Artikeln der deutschen Wikipedia) ein Markov-Modell und nutzt dieses, um
# zufällige neue Sätze zu generieren.
#
# Das Programm startet man so: python bot.py 6
# Dazu muss vorher noch das ZIP-Archiv corpus.zip entpackt werden (in dasselbe
# Verzeichnis, in dem auch dieses Programm liegt).
#
# Das Argument 6 gibt an, wie viele aufeinanderfolgende Wörter jeweils zu einem
# Zustand zusammengefasst werden sollen. Je höher diese Zahl ist, desto eher
# erhält man wörtliche Zitate aus dem Korpus. Dann sind die Sätze
# grammatikalisch einwandfrei, aber nicht besonders lustig. Je niedriger diese
# Zahl ist, desto eher weisen die generierten Sätze willkürliche Themensprünge
# auf. Damit erhält man lustigere Sätze.

from __future__ import division
import numpy as np
import random
import os
import sys
import cPickle as pickle

# In das Verzeichnis wechseln, in dem sich dieses Programm befindet.
os.chdir(sys.path[0])

DELIM = "#"

# Zieht aus einer gegebenen Wahrscheinlichkeitsverteilung einen Wert.
# Ist das Argument beispielsweise { 'a': 0.3, 'b': 0.6, 'c': 0.1 }, so wird mit
# 30 % Wahrscheinlichkeit ein 'a' gezogen, mit 60 % Wahrscheinlichkeit ein 'b'
# und mit 10 % Wahrscheinlichkeit ein 'c'.
#
# Die Verteilung muss nicht unbedingt normiert sein, das heißt die Summe der
# Werte in dem übergebenen Dictionary muss nicht unbedingt genau Eins sein.
def draw(d):
    u = np.random.uniform(0,1)
    v = 0
    s = sum(d.values())
    for k in d:
        v = v + d[k]/s
        if v >= u:
            return k

# Gegeben eine Markov-Kette `S` und ein initialer Zustand `x`, berechnet gemäß der
# Kette weitere Folgezustände. Je nach Wert der Flags `early` wird
# unterschiedlich abgebrochen: Ist `early` False, so wird erst abgebrochen,
# wenn das Ende eines Satzes erreicht wird. Ist `early` True, so wird schon
# abgebrochen, wenn der neue Zustand (ein Tupel aus mehreren Wörtern) irgendwo
# einen Satzpunkt enthält.
def forward(S, x, early=False):
    bs = []
    while True:
        bs.append(x)
        if early:
            if '.' in x:
                break
        else:
            if x[-1] == ".":
                break

        # `x` ist der aktuelle Zustand. `S[x]` gibt dann die
        # Wahrscheinlichkeitsverteilung für Folgezustände an.
        xs = S[x]
        if xs:
            x = draw(xs)
        # Wenn es keine Folgezustände gibt: abbrechen.
        else:
            break
    return bs

# Diese Funktion entnimmt einer Liste von Zuständen die
# Übergangswahrscheinlichkeiten und stellt eine Markov-Kette mit diesen
# Wahrscheinlichkeiten auf.
def learn(cs):
    D = {}
    for i in range(len(cs)-1):
        alt = cs[i]
        neu = cs[i+1]
        if alt in D:
            if neu in D[alt]:
                D[alt][neu] = D[alt][neu] + 1
            else:
                D[alt][neu] = 1
        else:
            D[alt] = { neu: 1 }
    return D

# Die folgende Funktion unterteilt einen Text (in der Variablen `corpus`) in
# eine Liste von aufeinanderfolgenden Wort-Tupeln der Länge n.
#
# Zum Beispiel wird bei n = 3 aus der Eingabe "Dies ist ein toller Test für
# Markov-Ketten" folgende Liste:
#
# [ "Dies#ist#ein", "ist#ein#toller", "ein#toller#Test", "toller#Test#für",
#   "Test#für#Markov-Ketten" ]
def group(corpus, n):
    alts = []
    cs   = []
    starts = {}
    for w in corpus.split(" ") + [""]:
        alts.append(w)
        if len(alts) == n:
            s = DELIM.join(alts)
            cs.append(s)
            alts.pop(0)
            if w in starts:
                starts[w].append(s)
            else:
                starts[w] = [s]
    return cs, starts

# Läuft die Markov-Kette `S` ab dem Zustand `x` beginnend vorwärts ab,
# läuft die Kette `T` ab demselben Zustand ebenfalls ab,
# und setzt die Ergebnisse zusammen.
#
# Das ist vor allem dann sinnvoll, wenn `T` aus demselben Korpus wie `S`
# entstanden ist, nur mit umgedrehter Wörterreihenfolge.
def forback(S,T, x):
    ws = forward(T, x, True)
    ws.reverse()
    ws.pop(0)
    if ws: ws.pop()

    us = forward(S,x)

    wws = []
    for w in ws + us:
        wws.append(w.split(DELIM)[0])
    wws = wws + us[-1].split(DELIM)[1:]
    return " ".join(wws)

cs     = None   # Der Korpus als Liste von Wörtertupeln
starts = None   # Zustände, die ein gegebenes Wort enthalten
S      = None   # Kette für die Vorwärtsentwicklung
T      = None   # Kette für die Rückwärtsentwicklung

N = int(sys.argv[1])  # Tupelgröße
try:
    cs, starts, S, T = pickle.load(open("corpus-" + sys.argv[1] + ".dat"))
except:
    print("Erzeuge Markov-Kette...")
    cs, starts = group(open("corpus.txt").read(), N)
    S  = learn(cs)
    cs.reverse()
    T  = learn(cs)
    pickle.dump((cs,starts,S,T), open("corpus-" + sys.argv[1] + ".dat", "w"))

while True:
    w = raw_input("> ")
    try:
        print(forback(S, T, random.choice(starts[w])))
    except:
        pass
