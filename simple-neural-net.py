#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import pickle
import time

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

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

    def train(self, samples, batchSize, eta, progress=None):
        samples = [] + samples

        v = self.cost(samples)

        while True:
            random.shuffle(samples)

            batches = [ samples[k:k+batchSize] for k in range(0,len(samples),batchSize) ]
            print("shuffle")
            for batch in batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for a, y in batch:
                    delta_nabla_b, delta_nabla_w = self.gradient(a, y)
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                try:
                    eta = float(open("eta.txt").read())
                    print("eta override!")
                except:
                    pass

                oldWeights = self.weights
                oldBiases  = self.biases
                self.weights = [w-(eta/len(batch))*nw 
                                for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(eta/len(batch))*nb 
                               for b, nb in zip(self.biases, nabla_b)]
                newV = self.cost(samples[0:1000])
                p = 0
                if progress: p = progress()
                if newV < v:
                    print("* %f@%f" % (v,eta), p)
                    v = newV
                    eta = eta * 1.2
                else:
                    print("  %f@%f" % (v,eta), p)
#                   self.weights = oldWeights
#                   self.biases  = oldBiases
                    v = newV
                    eta = eta / 2
#                   self.weights = [w-np.random.normal(0,0.2,w.shape[0]*w.shape[1]).reshape(w.shape[0],w.shape[1])
#                               for w, nw in zip(self.weights, nabla_w)]

    def cost(self, samples):
        return sum([ self.c(self.feedForward(a),np.array(y)) for a,y in samples ]) / len(samples)

    @classmethod
    def sigma(cls, z):
        return 1/(1+np.exp(-z))
#       return np.tanh(z)

    @classmethod
    def sigmaDeriv(cls, z):
        return cls.sigma(z) * (1 - cls.sigma(z))
#       return 1 - np.tanh(z)*np.tanh(z)

    @classmethod
    def c(cls, a, y):
        return 2 * np.linalg.norm(a - y)**2

    def cDeriv(cls, a, y):
        return a - y

net = Network([784,30,10])

training_data = training_data[0:10000]
test_data     = test_data[0:1000]
tr_data = training_data[0:1000]

def progress():
    train_results = [(np.argmax(net.feedForward(x)), np.argmax(y)) for (x, y) in tr_data]
    test_results = [(np.argmax(net.feedForward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in train_results), sum(int(x == y) for (x, y) in test_results)

try:
    net.train(training_data, 100, 0.001, progress)
except:
    print(time.time())
    pickle.dump((net.weights, net.biases), open("net-%d.pkl" % int(time.time()), "wb"))
