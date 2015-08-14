#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm simuliert das folgende neuronale Netzwerk für die vier
# Eingaben [0,0], [0,1], [1,0], [1,1].
#
#     *--+-----*
#         \   / \
#          \ /   \
#           /     *--
#          / \   /
#         /   \ /
#     *--+-----*
#
# Die Gewichte sind so gewählt, dass die XOR-Funktion berechnet wird.
# Die Ausgabe soll also bei den obigen Eingaben sein: 0, 1, 1, 0.
#
# Dieser Code verwendet Arrays.

import numpy as np

def sigma(t): return 1 / (1 + np.exp(-t))

V = np.array([[1000,-2000],[-2000,1000]])
W = np.array([[1000,1000]])
b = np.array([[-500],[-500]])
c = np.array([[-500]])

for x in [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]:
    y = sigma(np.dot(V,x) + b)
    z = sigma(np.dot(W,y) + c)
    print(z)
