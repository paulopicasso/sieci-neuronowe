import numpy as np
from math import e

def relu(x):
    # return np.minimum(np.maximum(x, 0), 100)
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + (1 / e ** x))


# def sigmoid(inputRow):
#     return np.array([1 / (1 + e**float(-x)) for x in inputRow], dtype='f8')


def reluDeriv(x):
    return np.minimum(np.maximum(np.ceil(x), 0), 1) 


def crossEntropy(outputs, expectedOutputs):
    length = np.size(outputs, 0)
    error = [expectedOutput * np.log(output) for expectedOutput, output in zip(expectedOutputs, outputs)]
    return (-1/length) * np.sum(error, 0)


def sigmoidDeriv(inputRow):
    sigmoidOutput = sigmoid(inputRow)
    return sigmoidOutput * (1 - sigmoidOutput)


def resolveDerivative(activationFunc):
    if(activationFunc == sigmoid):
        return sigmoidDeriv
    if(activationFunc == relu):
        return reluDeriv
    return sigmoidDeriv


def softmax(inputRow):
    exp = e ** inputRow
    return exp / np.sum(exp)


def initializeWeightsUniform(size, min, max):
    baseWeights = np.random.rand(size)
    return baseWeights * (max - min) + min


def heInitializeFilters(filterSize, numberOfFilters):
    baseWeights = np.random.randn(numberOfFilters, filterSize, filterSize)
    return baseWeights * np.sqrt(2 / (filterSize))

def xavierInitializeWeights(filterSize, numberOfFilters):
    baseWeights = np.random.randn(numberOfFilters, filterSize, filterSize)
    return baseWeights * np.sqrt(2 / (numberOfFilters + filterSize))