import numpy as np
from math import e

def relu(matrix):
    return np.minimum(np.maximum(matrix, 0), 10)


def sigmoid(inputRow):
    return np.array([1 / (1 + e**float(-x)) for x in inputRow], dtype='f8')


def crossEntropy(outputs, expectedOutputs):
    length = np.size(outputs, 0)
    error = [expectedOutput * np.log(output) for expectedOutput, output in zip(expectedOutputs, outputs)]
    return (-1/length) * np.sum(error, 0)


def sigmoidDeriv(inputRow):
    sigmoidOutput = sigmoid(inputRow)
    return sigmoidOutput * (1 - sigmoidOutput)


def reluDeriv(inputRow):
    return np.array([1 if x > 0 else 0 for x in inputRow])


def resolveDerivative(activationFunc):
    if(activationFunc == sigmoid):
        return sigmoidDeriv
    if(activationFunc == relu):
        return reluDeriv
    return sigmoidDeriv


def softmax(inputRow):
    exp = e ** inputRow
    return exp / np.sum(exp)


def heInitializeFilters(filterSize, numberOfFilters):
    baseWeights = np.random.randn(numberOfFilters, filterSize, filterSize)
    return baseWeights * np.sqrt(2 / (filterSize))
