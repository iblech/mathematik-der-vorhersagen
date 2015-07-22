#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

class Network():
    def __init__(self, layerSizes):
        self.initRandom(layerSizes)

    def initRandom(self, layerSizes):
        self.weights = []
        self.biases  = []

        for i in range(len(layerSizes) - 1):
            n, m = layerSizes[i], layerSizes[i+1]
            self.weights.append(np.random.normal(0, 1, n*m).reshape(m,n))
            self.biases .append(np.random.normal(0, 1, m)  .reshape(m,1))

    def feedForward(self, a):
        a = np.array(a).reshape(len(a), 1)
        for w, b in zip(self.weights, self.biases):
            a = self.sigma(np.dot(w,a) + b)
        return a

    def gradient(self, a, y):
        a = np.array(a).reshape(len(a), 1)
        y = np.array(y).reshape(len(y), 1)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs   = []
        acts = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            zs.append(z)
            a = self.sigma(z)
            acts.append(a)

        delta = self.cDeriv(acts[-1], y) * self.sigmaDeriv(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].T)
        for l in range(2, len(self.weights)+1):
            z   = zs[-l]
            spv = self.sigmaDeriv(z)
            delta = np.dot(self.weights[-l+1].T, delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, acts[-l-1].T)
        return (nabla_b, nabla_w)

    def train(self, samples, batchSize, eta):
        samples = [] + samples

        v = self.cost(samples)

        while True:
            random.shuffle(samples)

            batches = [ samples[k:k+batchSize] for k in range(0,len(samples),batchSize) ]
            for batch in batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for a, y in batch:
                    delta_nabla_b, delta_nabla_w = self.gradient(a, y)
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                oldWeights = self.weights
                oldBiases  = self.biases
                self.weights = [w-(eta/len(batch))*nw 
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(eta/len(batch))*nb 
                               for b, nb in zip(self.biases, nabla_b)]

                print(self.feedForward([1,2]))

                newV = self.cost(samples)
                if True or newV < v:
                    print("* %f@%f " % (v,eta))
                    v = newV
                    eta = eta * 1.2
                    eta = 5000
                    break
                else:
                    print("  %f@%f " % (v,eta))
                    self.weights = oldWeights
                    self.biases  = oldBiases
                    eta = eta / 2

    def cost(self, samples):
        return sum([ self.c(self.feedForward(a),np.array(y)) for a,y in samples ]) / len(samples)

    @classmethod
    def sigma(cls, z):
        return 1/(1+np.exp(-z))

    @classmethod
    def sigmaDeriv(cls, z):
        return cls.sigma(z) * (1 - cls.sigma(z))

    @classmethod
    def c(cls, a, y):
        return 2 * np.linalg.norm(a - y)**2

    def cDeriv(cls, a, y):
        return a - y

net = Network([2,10,10,2])

samples = []
for i in range(1000):
    a = np.random.normal(0,4,2)
    samples.append((a,a))

net.train(samples, 100, 1000000)
