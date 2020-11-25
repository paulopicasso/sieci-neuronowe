import numpy as np
import data_loader
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import neural_network
import helpers
import data_loader

inputSize = 28 * 28
outputSize = 10
minWeight = -0.0001
maxWeight = 0.0001
batchSize = 128
desiredError = 0.01
maxEpoch = 100
minDiff = 0.00001
alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
layerSizes = [1, 20, 50, 100, 500]

inputs = data_loader.loadTrainInputs()
labels = data_loader.loadTrainOutputs()

results = []
for layerSize in layerSizes:
    for alpha in alphas:

        avgLosses = []
        avgEpochs = []
        avgAcc = []
        for i in range(10):
            network = neural_network.NeuralNetwork(
                layerSize, inputSize, outputSize, minWeight, maxWeight, minWeight, maxWeight,
                helpers.relu
            )
            (epoch, accuracy) = neural_network.teachNeuralNetworkWithInput(
                network, batchSize, alpha, desiredError, maxEpoch, 10000, minDiff, inputs,
                labels
            )
            print('i: ', i)
            print(
                layerSize, alpha, epoch, accuracy
            )
            avgAcc.append(accuracy)
            avgEpochs.append(epoch)

        epoch = np.average(avgEpochs)
        accuracy = np.average(avgAcc)
        
        result = (layerSize, alpha, epoch, accuracy)
        results.append(result)