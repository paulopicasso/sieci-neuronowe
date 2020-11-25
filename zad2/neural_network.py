import numpy as np
import data_loader
from math import e
import helpers

class NeuralNetwork:

    def __init__(self, hiddenLayerSize, inputSize, outputSize, minWeight, maxWeight, minBias, maxBias, activationFun):
        self.hiddenLayerWeights = self.initializeWeights(hiddenLayerSize, inputSize, minWeight, maxWeight)
        self.hiddenLayerBiases = self.initializeBiases(hiddenLayerSize, minBias, maxBias)
        self.outputLayerWeights = self.initializeWeights(outputSize, hiddenLayerSize, minWeight, maxWeight)
        self.outputLayerBiases = self.initializeBiases(outputSize, minBias, maxBias)
        self.hiddenLayerActivations = []
        self.hiddenLayerErrors = []
        self.hiddenLayerNetValues = np.array([])
        self.outputLayerActivations = []
        self.outputLayerErrors = []
        self.outputLayerNetValues = np.array([])
        self.activationFun = activationFun
        self.activationFunDeriv = helpers.resolveDerivative(self.activationFun)
        

    def initializeBiases(self, numberOfNeurons, min, max):
        baseBiases = np.random.rand(numberOfNeurons)
        return baseBiases * (max - min) + min


    def initializeWeights(self, numberOfNeurons, inputSize, min, max):
        baseWeights = np.random.rand(numberOfNeurons, inputSize)
        return baseWeights * (max - min) + min


    def feedForward(self, inputRow):
        hiddenNet = (self.hiddenLayerWeights @ inputRow) + self.hiddenLayerBiases
        hiddenActivation = self.activationFun(hiddenNet)
        outputNet = (self.outputLayerWeights @ hiddenActivation) + self.outputLayerBiases
        outputActivation = helpers.softmax(outputNet)
        
        self.hiddenLayerNetValues = hiddenNet
        self.hiddenLayerActivations.append(hiddenActivation)
        self.outputLayerNetValues = outputNet
        self.outputLayerActivations.append(outputActivation)

        return outputActivation


    def backPropagateError(self, output, expectedOutput):
        outputLayerError = expectedOutput - output
        hiddenLayerError = (self.outputLayerWeights.T @ outputLayerError) * self.activationFunDeriv(self.hiddenLayerNetValues)

        self.outputLayerErrors.append(outputLayerError)
        self.hiddenLayerErrors.append(hiddenLayerError)


    def updateWeights(self, learningRate, batch):
        batchSize = np.size(batch)

        outputLayerDelta = np.array(self.outputLayerErrors).T @ np.array(self.hiddenLayerActivations)
        hiddenLayerDelta = np.array(self.hiddenLayerErrors).T @ batch

        self.outputLayerWeights = self.outputLayerWeights + (learningRate / batchSize) * outputLayerDelta
        self.hiddenLayerWeights = self.hiddenLayerWeights + (learningRate / batchSize) * hiddenLayerDelta
        self.outputLayerBiases = self.outputLayerBiases + (learningRate / batchSize) * np.sum(self.outputLayerErrors, 0)
        self.hiddenLayerBiases = self.hiddenLayerBiases + (learningRate / batchSize) * np.sum(self.hiddenLayerErrors, 0)


    def clearLayers(self):
        self.hiddenLayerActivations = []
        self.hiddenLayerErrors = []
        self.hiddenLayerNetValues = np.array([])
        self.outputLayerActivations = []
        self.outputLayerErrors = []
        self.outputLayerNetValues = np.array([])


    def validateModel(self, inputs, expectedOutputs):
        outputs = [self.feedForward(inputRow) for inputRow in inputs]
        results = [np.argmax(output) for output in outputs]

        correct = 0
        for i, result in enumerate(results):
            if(expectedOutputs[i][result] == 1):
                correct = correct + 1

        validationSize = np.size(inputs, 0)
        print('Correct: ', correct, '/', validationSize)

        return (np.average(helpers.crossEntropy(outputs, expectedOutputs)), correct / validationSize)


    def getLastDiffs(self, errList, noOfElements):
        diffs = [
            li - lj for li, lj
            in zip(errList[-noOfElements:-1], errList[-noOfElements + 1:len(errList)])
        ]
        return diffs


    def learn(self, inputs, labels, batchSize, learningRate, validationInputs, validationLabels, desiredError, maxEpoch, minDiff):
        error = desiredError
        epoch = 1
        diff = minDiff
        losses = []

        while error >= desiredError and epoch <= maxEpoch and diff >= minDiff:
            print('Epoch: ', epoch)

            batchIndex = 0
            while batchIndex + batchSize <= np.size(inputs, 0): # Iterating batches
                nextBatchIndex = batchIndex + batchSize
                batch = inputs[batchIndex:nextBatchIndex]

                for i, inputRow in enumerate(batch): # Processing one batch
                    output = self.feedForward(inputRow)
                    expectedOutput = labels[batchIndex + i]
                    self.backPropagateError(output, expectedOutput)
                    
                self.updateWeights(learningRate, batch)
                self.clearLayers()
                batchIndex = nextBatchIndex

            
            (error, accuracy) = self.validateModel(validationInputs, validationLabels)
            losses.append(error)
            print('Error: ', error)

            self.clearLayers()
            epoch = epoch + 1
            diff = max(self.getLastDiffs(losses, 5)) if len(losses) >= 5 else minDiff 

        return (epoch - 1, accuracy)
            

def teachNeuralNetworkWithInput(
    network: NeuralNetwork,
    batchSize: int,
    learningRate: float,
    desiredError: float,
    maxEpoch: int,
    validationSetSize: int,
    minDiff: float,
    inputs: np.ndarray,
    labels: np.ndarray
):
    dataSize = np.size(inputs, 0)
    trainInputs = inputs[0:-validationSetSize]
    trainLabels = labels[0:-validationSetSize]
    validationInputs = inputs[-validationSetSize:dataSize]
    validationLabels = labels[-validationSetSize:dataSize]

    return network.learn(
        trainInputs,
        trainLabels,
        batchSize,
        learningRate,
        validationInputs,
        validationLabels,
        desiredError,
        maxEpoch,
        minDiff
    )


# hiddenLayerSize = 100
# inputSize = 28 * 28
# outputSize = 10
# minweight = -0.0001
# maxweight = 0.0001
# minbias = -0.001
# maxbias = 0.001
# batchSize = 20
# learningRate = 0.01
# desiredError = 0.01
# maxEpoch = 100
# validationSetSize = 1000
# minDiff = 0.0001

# inputs = data_loader.loadTrainInputs()
# labels = data_loader.loadTrainOutputs()

# network = NeuralNetwork(
#     hiddenLayerSize, inputSize, outputSize,
#     minweight, maxweight, minbias, maxbias,
#     helpers.relu
# )

# print('Learning...')
# teachNeuralNetworkWithInput(
#     network,
#     batchSize,
#     learningRate,
#     desiredError,
#     maxEpoch,
#     validationSetSize,
#     minDiff,
#     inputs,
#     labels
# )
print('Finished.')
