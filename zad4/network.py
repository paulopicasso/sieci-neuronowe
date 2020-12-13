import numpy as np
import helpers
import neural_network as mlp
import data_loader
import mlp_helpers

class Network:

    def __init__(
        self,
        paddingSize = 1,
        step = 1,
        filterSize = 3,
        numberOfFilters = 32,
        inputSize = 28,
        featureMapSize = 28,
        activationFun = helpers.relu,
        activationDeriv = helpers.reluDeriv,
        poolingWindowSize = 2,
        hiddenLayerSize = 128,
        weightMin = -0.001,
        weightMax = 0.001,
        weightInitializer = helpers.heInitializeFilters
    ):
        self.weightInitializer = weightInitializer
        self.filterTensor = self.initializeFilterTensor(filterSize, numberOfFilters)
        self.featureMapTensor = np.empty((numberOfFilters, featureMapSize, featureMapSize))
        self.featureMapNetTensor = np.empty(self.featureMapTensor.shape)
        self.poolingFeatureMapTensor = np.empty((numberOfFilters, int(featureMapSize / poolingWindowSize), int(featureMapSize / poolingWindowSize)))
        self.poolingIndicesTensorX = np.empty(self.poolingFeatureMapTensor.shape)
        self.poolingIndicesTensorY = np.empty(self.poolingFeatureMapTensor.shape)
        self.poolingIndicesTensorZ = np.empty(self.poolingFeatureMapTensor.shape)
        self.mlpNetwork = mlp.NeuralNetwork(
            hiddenLayerSize,
            numberOfFilters * int(featureMapSize / poolingWindowSize) * int(featureMapSize / poolingWindowSize),
            10,
            -0.1, 0.1, -0.1, 0.1,
            mlp_helpers.relu
        )
        self.layer3Weights = helpers.initializeWeightsUniform(
            numberOfFilters * int(featureMapSize / poolingWindowSize) * int(featureMapSize / poolingWindowSize),
            weightMin,
            weightMax
        )
        self.poolingErrorsTensor = np.empty(self.featureMapTensor.shape)
        self.convolutionErrorsTensor = np.empty(self.featureMapTensor.shape)
        self.convolutionErrorsFilters = np.zeros(self.filterTensor.shape)
        self.layer3Net = []
        self.layer3Activation = []
        self.layer3Errors = []
        self.paddingSize = paddingSize
        self.step = step
        self.filterSize = filterSize
        self.numberOfFilters = numberOfFilters
        self.inputSize = inputSize
        self.featureMapSize = featureMapSize
        self.activationFun = activationFun
        self.activationDeriv = activationDeriv
        self.poolingWindowSize = poolingWindowSize
        
    
    def initializeFilterTensor(self, filterSize, numberOfFilters):
        return self.weightInitializer(filterSize, numberOfFilters)
        # return np.zeros((numberOfFilters, filterSize, filterSize)) + 1


    def prepareImageForConvolution(self, image):
        paddedImage = np.pad(image, self.paddingSize)
        (height, width) = paddedImage.shape
        imageTensor = np.reshape(paddedImage, (1, height, width))
        return np.repeat(imageTensor, self.numberOfFilters, 0)


    def convolute(self, image):
        inputTensor = self.prepareImageForConvolution(image)

        for i in range(self.inputSize):
            for j in range(self.inputSize):
                inputSlice = inputTensor[:, i:i + self.filterSize, j:j + self.filterSize]
                mult = inputSlice * self.filterTensor
                net = np.sum(mult, (1,2))

                self.featureMapNetTensor[:, i, j] = net

        self.featureMapTensor = self.activationFun(self.featureMapNetTensor)


    def pool(self):
        for i in range(self.featureMapSize)[::self.poolingWindowSize]:
            for j in range(self.featureMapSize)[::self.poolingWindowSize]:
                featureMapSlice = self.featureMapTensor[:, i:i + self.poolingWindowSize, j:j + self.poolingWindowSize]
                
                poolingMapI = int(i / self.poolingWindowSize)
                poolingMapJ = int(j / self.poolingWindowSize)
                self.poolingFeatureMapTensor[:, poolingMapI, poolingMapJ] = np.max(featureMapSlice, (1,2))

                indices = [
                    tuple([i]) + np.unravel_index(np.argmax(x), (self.poolingWindowSize,self.poolingWindowSize))
                    for i, x in enumerate(featureMapSlice)
                ]
                self.poolingIndicesTensorX[:, poolingMapI, poolingMapJ] = [x for x, _, _ in indices]
                self.poolingIndicesTensorY[:, poolingMapI, poolingMapJ] = [x for _, x, _ in indices]
                self.poolingIndicesTensorZ[:, poolingMapI, poolingMapJ] = [x for _, _, x in indices]


    def flatten(self):
        flatPooling = self.poolingFeatureMapTensor.flatten()
        net = flatPooling * self.layer3Weights
        
        self.layer3Net = net
        self.layer3Activation = self.activationFun(net)


    def calculateLayer3Error(self):
        part1 = self.mlpNetwork.hiddenLayerWeights.T @ self.mlpNetwork.hiddenLayerErrors[-1]
        self.layer3Errors = part1 * self.activationDeriv(self.layer3Activation)


    def calculatePoolingError(self):
        indicesX = self.poolingIndicesTensorX.flatten()
        indicesY = self.poolingIndicesTensorY.flatten()
        indicesZ = self.poolingIndicesTensorZ.flatten()
        for error, ix, iy, iz in zip(self.layer3Errors, indicesX, indicesY, indicesZ):
            self.poolingErrorsTensor[int(ix), int(iy), int(iz)] = error 


    def calculateConvolutionError(self):
        self.convolutionErrorsTensor = self.poolingErrorsTensor * self.activationDeriv(self.featureMapNetTensor)


    def propagateErrorsToFilters(self):
        convolutionErrorsPadded = np.pad(self.convolutionErrorsTensor, ((0,0), (1,1), (1,1)))
        for i in range(self.featureMapSize):
            for j in range(self.featureMapSize):
                errorSlice = convolutionErrorsPadded[:, i:i + self.filterSize, j:j + self.filterSize]
                self.convolutionErrorsFilters += errorSlice
                #TODO padding?

        size = self.featureMapSize * self.featureMapSize
        # self.convolutionErrorsFilters /= size


    def updateConvolutionWeights(self, image, learningRate):
        gradients = np.zeros(self.convolutionErrorsFilters.shape)
        rotatedImageTensor = self.prepareImageForConvolution(np.rot90(image, 2))
        for i in range(self.inputSize):
            for j in range(self.inputSize):
                imageSlice = rotatedImageTensor[:, i:i + self.filterSize, j:j + self.filterSize]
                gradients += self.convolutionErrorsFilters * imageSlice

        self.filterTensor += learningRate * gradients


    def updateLayer3Weights(self, learningRate):
        gradient = self.layer3Errors.T @ self.poolingFeatureMapTensor.flatten()
        self.layer3Weights += learningRate * gradient


    def updateMLPWeights(self, learningRate, input, step):
        self.mlpNetwork.updateWeights(learningRate, np.reshape(input, (1, -1)), step)


    def feedForward(self, image):
        self.convolute(image)
        self.pool()
        # self.flatten()
        self.layer3Activation = self.poolingFeatureMapTensor.flatten()
        return self.mlpNetwork.feedForward(self.layer3Activation)


    def backpropagateError(self, expectedOutput):
        self.mlpNetwork.backPropagateError(self.mlpNetwork.outputLayerActivations[-1], expectedOutput)
        # self.calculateLayer3Error()
        self.layer3Errors = self.mlpNetwork.hiddenLayerErrors[-1]
        self.calculatePoolingError()
        self.calculateConvolutionError()
        self.propagateErrorsToFilters()

    
    def updateWeights(self, image,convolutionLearningRate, learningRate, step):
        self.updateConvolutionWeights(image, convolutionLearningRate)
        # self.updateLayer3Weights(learningRate)
        self.updateMLPWeights(learningRate, self.layer3Activation, step)


    def validateModel(self, validationInputs, validationLabels):
        outputs = [self.feedForward(image) for image in validationInputs]
        results = [np.argmax(output) for output in outputs]

        correct = 0
        for i, result in enumerate(results):
            if(validationLabels[i][result] == 1):
                correct += 1

        validationSize = np.size(validationInputs, 0)
        print('Correct: ', correct, '/', validationSize)

        return (np.average(helpers.crossEntropy(outputs, validationLabels)), correct / validationSize)


    def getLastDiffs(self, errList, noOfElements):
        diffs = [
            li - lj for li, lj
            in zip(errList[-noOfElements:-1], errList[-noOfElements + 1:len(errList)])
        ]
        return diffs


    def learn(self, convolutionLearningRate, learningRate, desiredError, maxEpoch, minDiff, validationSetSize=10000):
        inputs = data_loader.loadTrainInputs()
        labels = data_loader.loadTrainOutputs()
        dataSize = np.size(inputs, 0)
        validationInputs = inputs[-validationSetSize:dataSize]
        validationLabels = labels[-validationSetSize:dataSize]
        inputs = inputs[0:-validationSetSize]
        labels = labels[0:-validationSetSize]

        error = desiredError
        epoch = 1
        step = 1
        diff = minDiff
        losses = []
        accuracies = []

        while error >= desiredError and epoch <= maxEpoch and diff >= minDiff:
            print('Epoch: ', epoch)

            for i, image in enumerate(inputs): 
                self.feedForward(image)
                expectedOutput = labels[i]
                self.backpropagateError(expectedOutput)
                self.updateWeights(image, convolutionLearningRate, learningRate, step)
                self.mlpNetwork.clearLayers()

                if step % 5000 == 0:
                    (error, accuracy) = self.validateModel(validationInputs, validationLabels)
                    print('Error: ', error)
                    self.mlpNetwork.clearLayers()

                step += 1
            
            (error, accuracy) = self.validateModel(validationInputs, validationLabels)
            losses.append(error)
            accuracies.append(accuracy)
            print('Error: ', error)
            self.mlpNetwork.clearLayers()

            epoch += 1
            diff = max(self.getLastDiffs(losses, 5)) if len(losses) >= 5 else minDiff 

        return (epoch - 1, accuracies)
