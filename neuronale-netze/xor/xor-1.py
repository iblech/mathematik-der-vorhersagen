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
# Dieser Code verwendet nur einfache Zahlvariablen, keine Arrays.

import numpy as np

def sigma(t): return 1 / (1 + np.exp(-t))

v00 = +1000
v01 = -2000
v10 = -2000
v11 = +1000
b0  = -500
b1  = -500

w00 = 1000
w01 = 1000
c0  = -500

for x0, x1 in [(0,0), (0,1), (1,0), (1,1)]:
    y0 = sigma(v00*x0 + v01*x1 + b0)
    y1 = sigma(v10*x0 + v11*x1 + b1)
    z0 = sigma(w00*y0 + w01*y1 + c0)
    print(z0)
