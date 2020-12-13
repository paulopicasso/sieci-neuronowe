import numpy as np
import mlp_data_loader as data_loader
from math import e
import mlp_helpers as helpers

class NeuralNetwork:

    def __init__(
        self,
        hiddenLayerSize,
        inputSize,
        outputSize,
        minWeight,
        maxWeight,
        minBias,
        maxBias,
        activationFun,
        useMomentum = False,
        momentumRate = 0,
        useNag = False,
        weightInitializer = helpers.normalInitializeWeights,
        learningRateMode = 'standard'
    ):
        self.weightInitializer = weightInitializer
        self.hiddenLayerWeights = self.initializeWeights(hiddenLayerSize, inputSize, minWeight, maxWeight)
        self.hiddenLayerBiases = self.initializeBiases(hiddenLayerSize, minBias, maxBias)
        self.outputLayerWeights = self.initializeWeights(outputSize, hiddenLayerSize, minWeight, maxWeight)
        self.outputLayerBiases = self.initializeBiases(outputSize, minBias, maxBias)
        self.hiddenLayerActivations = []
        self.hiddenLayerErrors = []
        self.hiddenLayerNetValues = np.array([])
        self.hiddenNagNetValues = np.array([])
        self.outputLayerActivations = []
        self.outputNagActivation = np.array([])
        self.outputLayerErrors = []
        self.outputLayerNetValues = np.array([])
        self.activationFun = activationFun
        self.activationFunDeriv = helpers.resolveDerivative(self.activationFun)
        self.momentumRate = momentumRate
        self.outputMomentum = 0
        self.hiddenMomentum = 0
        self.outputBiasMomentum = 0
        self.hiddenBiasMomentum = 0
        self.useMomentum = useMomentum
        self.useNag = useNag
        self.hiddenGradientSum = 0
        self.outputGradientSum = 0
        self.hiddenBiasGradientSum = 0
        self.outputBiasGradientSum = 0
        self.hiddenGradientAvg = 0
        self.outputGradientAvg = 0
        self.hiddenBiasGradientAvg = 0
        self.outputBiasGradientAvg = 0
        self.hiddenDeltaAvg = 0
        self.outputDeltaAvg = 0
        self.hiddenBiasDeltaAvg = 0
        self.outputBiasDeltaAvg = 0
        self.outputMt = 0
        self.hiddenMt = 0
        self.outputVt = 0
        self.hiddenVt = 0
        self.outputBiasMt = 0
        self.hiddenBiasMt = 0
        self.outputBiasVt = 0
        self.hiddenBiasVt = 0
        self.outputMtC = 0
        self.hiddenMtC = 0
        self.outputVtC = 0
        self.hiddenVtC = 0
        self.outputBiasMtC = 0
        self.hiddenBiasMtC = 0
        self.outputBiasVtC = 0
        self.hiddenBiasVtC = 0
        self.learningRateMode = learningRateMode
        self.epsilon = 10e-6
        self.gamma = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        

    def initializeBiases(self, numberOfNeurons, min, max):
        baseBiases = np.random.randn(numberOfNeurons)
        return baseBiases * (max - min) + min


    def initializeWeights(self, numberOfNeurons, inputSize, min, max):
        return self.weightInitializer(numberOfNeurons, inputSize, min, max)


    def feedForward(self, inputRow):
        hiddenNet = (self.hiddenLayerWeights @ inputRow) + self.hiddenLayerBiases
        hiddenActivation = self.activationFun(hiddenNet)
        outputNet = (self.outputLayerWeights @ hiddenActivation) + self.outputLayerBiases
        outputActivation = helpers.softmax(outputNet)
        
        self.hiddenLayerNetValues = hiddenNet
        self.hiddenLayerActivations.append(hiddenActivation)
        self.outputLayerNetValues = outputNet
        self.outputLayerActivations.append(outputActivation)

        if self.useNag:
            hiddenNagNet = ((self.hiddenLayerWeights - self.hiddenMomentum) @ inputRow) + (self.hiddenLayerBiases - self.hiddenBiasMomentum)
            outputNagNet = ((self.outputLayerWeights - self.outputMomentum) @ hiddenActivation) + (self.outputLayerBiases - self.outputBiasMomentum)
            outputNagActivation = helpers.softmax(outputNagNet)
            self.hiddenNagNetValues = hiddenNagNet
            self.outputNagActivation = outputNagActivation

        return outputActivation


    def backPropagateError(self, output, expectedOutput):
        outputLayerError = expectedOutput - output
        hiddenLayerError = (self.outputLayerWeights.T @ outputLayerError) * self.activationFunDeriv(self.hiddenLayerNetValues)

        self.outputLayerErrors.append(outputLayerError)
        self.hiddenLayerErrors.append(hiddenLayerError)


    def updateAdagradGradients(self, hiddenGradient, outputGradient, hiddenBiasGradient, outputBiasGradient):
        self.hiddenGradientSum += np.square(hiddenGradient)
        self.outputGradientSum += np.square(outputGradient)
        self.hiddenBiasGradientSum += np.square(hiddenBiasGradient)
        self.outputBiasGradientSum += np.square(outputBiasGradient)

    
    def updateAdadeltaDeltas(self, hiddenDelta, outputDelta, hiddenBiasDelta, outputBiasDelta):
        self.hiddenDeltaAvg = self.hiddenDeltaAvg * self.gamma + (1 - self.gamma) * np.square(hiddenDelta)
        self.outputDeltaAvg = self.outputDeltaAvg * self.gamma + (1 - self.gamma) * np.square(outputDelta)
        self.hiddenBiasDeltaAvg = self.hiddenBiasDeltaAvg * self.gamma + (1 - self.gamma) * np.square(hiddenBiasDelta)
        self.outputBiasDeltaAvg = self.outputBiasDeltaAvg * self.gamma + (1 - self.gamma) * np.square(outputBiasDelta)


    def updateAdadeltaGradients(self, hiddenGradient, outputGradient, hiddenBiasGradient, outputBiasGradient):
        self.hiddenGradientAvg = self.hiddenGradientAvg * self.gamma + (1 - self.gamma) * np.square(hiddenGradient)
        self.outputGradientAvg = self.outputGradientAvg * self.gamma + (1 - self.gamma) * np.square(outputGradient)
        self.hiddenBiasGradientAvg = self.hiddenBiasGradientAvg * self.gamma + (1 - self.gamma) * np.square(hiddenBiasGradient)
        self.outputBiasGradientAvg = self.outputBiasGradientAvg * self.gamma + (1 - self.gamma) * np.square(outputBiasGradient)


    def updateAdam(self, hiddenGradient, outputGradient, hiddenBiasGradient, outputBiasGradient, step):
        self.hiddenMt = self.beta1 * self.hiddenMt + (1 - self.beta1) * hiddenGradient
        self.outputMt = self.beta1 * self.outputMt + (1 - self.beta1) * outputGradient
        self.hiddenBiasMt = self.beta1 * self.hiddenBiasMt + (1 - self.beta1) * hiddenBiasGradient
        self.outputBiasMt = self.beta1 * self.outputBiasMt + (1 - self.beta1) * outputBiasGradient

        self.hiddenVt = self.beta2 * self.hiddenVt + (1 - self.beta2) * np.square(hiddenGradient)
        self.outputVt = self.beta2 * self.outputVt + (1 - self.beta2) * np.square(outputGradient)
        self.hiddenBiasVt = self.beta2 * self.hiddenBiasVt + (1 - self.beta2) * np.square(hiddenBiasGradient)
        self.outputBiasVt = self.beta2 * self.outputBiasVt + (1 - self.beta2) * np.square(outputBiasGradient)

        self.hiddenMtC = self.hiddenMt / (1 - self.beta1 ** step)
        self.outputMtC = self.outputMt / (1 - self.beta1 ** step)
        self.hiddenBiasMtC = self.hiddenBiasMt / (1 - self.beta1 ** step)
        self.outputBiasMtC = self.outputBiasMt / (1 - self.beta1 ** step)

        self.hiddenVtC = self.hiddenVt / (1 - self.beta2 ** step)
        self.outputVtC = self.outputVt / (1 - self.beta2 ** step)
        self.hiddenBiasVtC = self.hiddenBiasVt / (1 - self.beta2 ** step)
        self.outputBiasVtC = self.outputBiasVt / (1 - self.beta2 ** step)


    def backPropagateErrorNag(self, expectedOutput):
        outputLayerError = expectedOutput - self.outputNagActivation
        hiddenLayerError = (self.outputLayerWeights.T @ outputLayerError) * self.activationFunDeriv(self.hiddenNagNetValues)

        self.outputLayerErrors.append(outputLayerError)
        self.hiddenLayerErrors.append(hiddenLayerError)


    def updateWeights(self, learningRate, batch, step):
        batchSize = np.size(batch)

        outputLayerGradient = np.array(self.outputLayerErrors).T @ np.array(self.hiddenLayerActivations)
        hiddenLayerGradient = np.array(self.hiddenLayerErrors).T @ batch
        outputBiasGradient = np.sum(self.outputLayerErrors, 0)
        hiddenBiasGradient = np.sum(self.hiddenLayerErrors, 0)

        if self.learningRateMode == 'adadelta':
            self.updateAdadeltaGradients(hiddenLayerGradient, outputLayerGradient, hiddenBiasGradient, outputBiasGradient)

        if self.learningRateMode == 'adagrad':
            self.updateAdagradGradients(hiddenLayerGradient, outputLayerGradient, hiddenBiasGradient, outputBiasGradient)

        if self.learningRateMode == 'adam':
            self.updateAdam(hiddenLayerGradient, outputLayerGradient, hiddenBiasGradient, outputBiasGradient, step)

        outputDeltaModifier = {
            'standard': learningRate / batchSize,
            'adagrad': learningRate / np.sqrt(self.outputGradientSum + self.epsilon),
            'adadelta': np.sqrt(self.outputDeltaAvg + self.epsilon) / np.sqrt(self.outputGradientAvg + self.epsilon),
            'adam': learningRate / (np.sqrt(self.outputVtC) + self.epsilon)
        }[self.learningRateMode]

        hiddenDeltaModifier = {
            'standard': learningRate / batchSize,
            'adagrad': learningRate / np.sqrt(self.hiddenGradientSum + self.epsilon), 
            'adadelta': np.sqrt(self.hiddenDeltaAvg + self.epsilon) / np.sqrt(self.hiddenGradientAvg + self.epsilon), 
            'adam': learningRate / (np.sqrt(self.hiddenVtC) + self.epsilon) 
        }[self.learningRateMode]

        outputBiasDeltaModifier = {
            'standard': learningRate / batchSize,
            'adagrad': learningRate / np.sqrt(self.outputBiasGradientSum + self.epsilon), 
            'adadelta': np.sqrt(self.outputBiasDeltaAvg + self.epsilon) / np.sqrt(self.outputBiasGradientAvg + self.epsilon), 
            'adam': learningRate / (np.sqrt(self.outputBiasVtC) + self.epsilon) 
        }[self.learningRateMode]

        hiddenBiasDeltaModifier = {
            'standard': learningRate / batchSize,
            'adagrad': learningRate / np.sqrt(self.hiddenBiasGradientSum + self.epsilon), 
            'adadelta': np.sqrt(self.hiddenBiasDeltaAvg + self.epsilon) / np.sqrt(self.hiddenBiasGradientAvg + self.epsilon), 
            'adam': learningRate / (np.sqrt(self.hiddenBiasVtC) + self.epsilon) 
        }[self.learningRateMode]

        outputLayerGradient = -self.outputMtC if learningRate == 'adam' else outputLayerGradient
        hiddenLayerGradient = -self.hiddenMtC if learningRate == 'adam' else hiddenLayerGradient
        outputBiasGradient = -self.outputBiasMtC if learningRate == 'adam' else outputBiasGradient
        hiddenBiasGradient = -self.hiddenBiasMtC if learningRate == 'adam' else hiddenBiasGradient

        outputLayerDelta = outputDeltaModifier * outputLayerGradient + self.outputMomentum
        hiddenLayerDelta = hiddenDeltaModifier * hiddenLayerGradient + self.hiddenMomentum
        outputLayerBiasDelta = outputBiasDeltaModifier * outputBiasGradient + self.outputBiasMomentum
        hiddenLayerBiasDelta = hiddenBiasDeltaModifier * hiddenBiasGradient + self.hiddenBiasMomentum

        if self.learningRateMode == 'adadelta':
            self.updateAdadeltaDeltas(hiddenLayerDelta, outputLayerDelta, hiddenLayerBiasDelta, outputLayerBiasDelta)

        if self.useMomentum:
            self.outputMomentum = self.momentumRate * outputLayerDelta
            self.hiddenMomentum = self.momentumRate * hiddenLayerDelta
            self.outputBiasMomentum = self.momentumRate * outputLayerBiasDelta
            self.hiddenBiasMomentum = self.momentumRate * hiddenLayerBiasDelta

        self.outputLayerWeights = self.outputLayerWeights + outputLayerDelta
        self.hiddenLayerWeights = self.hiddenLayerWeights + hiddenLayerDelta
        self.outputLayerBiases = self.outputLayerBiases + outputLayerBiasDelta
        self.hiddenLayerBiases = self.hiddenLayerBiases + hiddenLayerBiasDelta


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
        step = 1
        diff = minDiff
        losses = []
        accuracies = []

        while error >= desiredError and epoch <= maxEpoch and diff >= minDiff:
            print('Epoch: ', epoch)

            batchIndex = 0
            while batchIndex + batchSize <= np.size(inputs, 0): # Iterating batches
                nextBatchIndex = batchIndex + batchSize
                batch = inputs[batchIndex:nextBatchIndex]

                for i, inputRow in enumerate(batch): # Processing one batch
                    output = self.feedForward(inputRow)
                    expectedOutput = labels[batchIndex + i]
                    if self.useNag:
                        self.backPropagateErrorNag(expectedOutput)
                    else:
                        self.backPropagateError(output, expectedOutput)
                    
                self.updateWeights(learningRate, batch, step)
                self.clearLayers()
                batchIndex = nextBatchIndex
                step += 1

            
            (error, accuracy) = self.validateModel(validationInputs, validationLabels)
            losses.append(error)
            accuracies.append(accuracy)
            print('Error: ', error)

            self.clearLayers()
            epoch += 1
            diff = max(self.getLastDiffs(losses, 5)) if len(losses) >= 5 else minDiff 

        return (epoch - 1, accuracies)
            

def teachNeuralNetworkWithInput(
    network: NeuralNetwork,
    batchSize: int,
    learningRate: float,
    desiredError: float,
    maxEpoch: int,
    validationSetSize: int,
    minDiff: float,
    inputs: np.ndarray,
    labels: np.ndarray,
    trainSize = 0,
):
    dataSize = np.size(inputs, 0)

    if trainSize == 0:
        trainInputs = inputs[0:-validationSetSize]
        trainLabels = labels[0:-validationSetSize]
        validationInputs = inputs[-validationSetSize:dataSize]
        validationLabels = labels[-validationSetSize:dataSize]
    else:
        trainInputs = inputs[0:trainSize]
        trainLabels = labels[0:trainSize]
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

