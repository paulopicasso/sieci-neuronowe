import numpy as np
from math import e

# def relu(inputRow):
#     return np.array([min(max(0, x), 10) for x in inputRow])
def relu(x):
    return np.minimum(np.maximum(x, 0), 10)
    # return np.maximum(x, 0)


def sigmoid(inputRow):
    return np.array([1 / (1 + e**float(-x)) for x in inputRow], dtype='f8')


def crossEntropy(outputs, expectedOutputs):
    length = np.size(outputs, 0)
    error = [expectedOutput * np.log(output) for expectedOutput, output in zip(expectedOutputs, outputs)]
    return (-1/length) * np.sum(error, 0)


def sigmoidDeriv(inputRow):
    sigmoidOutput = sigmoid(inputRow)
    return sigmoidOutput * (1 - sigmoidOutput)


def reluDeriv(x):
    return np.minimum(np.maximum(np.ceil(x), 0), 1) 



def resolveDerivative(activationFunc):
    if(activationFunc == sigmoid):
        return sigmoidDeriv
    if(activationFunc == relu):
        return reluDeriv
    return sigmoidDeriv


def softmax(inputRow):
    exp = e ** inputRow
    return exp / np.sum(exp)


def normalInitializeWeights(numberOfNeurons, inputSize, min, max):
    baseWeights = np.random.rand(numberOfNeurons, inputSize)
    return baseWeights * (max - min) + min


def xavierInitializeWeights(numberOfNeurons, inputSize, min, max):
    baseWeights = np.random.randn(numberOfNeurons, inputSize)
    return baseWeights * np.sqrt(2 / (numberOfNeurons + inputSize))


def heInitializeWeights(numberOfNeurons, inputSize, min, max):
    baseWeights = np.random.randn(numberOfNeurons, inputSize)
    return baseWeights * np.sqrt(2 / (inputSize))