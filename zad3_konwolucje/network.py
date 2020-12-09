import numpy as np
import helpers
import mlp_network.neural_network as mlp

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
        poolingWindowSize = 2,
        hiddenLayerSize = 128
    ):
        self.filterTensor = self.initializeFilterTensor(filterSize, numberOfFilters)
        self.featureMapTensor = np.empty((numberOfFilters, featureMapSize, featureMapSize))
        self.poolingFeatureMapTensor = np.empty((numberOfFilters, featureMapSize / poolingWindowSize, featureMapSize / poolingWindowSize))
        self.mlpNetwork = mlp.NeuralNetwork(
            hiddenLayerSize,
            numberOfFilters * (featureMapSize / poolingWindowSize),
            10,
            -0.1, 0.1, -0.1, 0.1,
            activationFun
        )
        self.paddingSize = paddingSize
        self.step = step
        self.filterSize = filterSize
        self.numberOfFilters = numberOfFilters
        self.inputSize = inputSize
        self.featureMapSize = featureMapSize
        self.activationFun = activationFun
        self.poolingWindowSize = poolingWindowSize
        
    
    def initializeFilterTensor(self, filterSize, numberOfFilters):
        return helpers.heInitializeFilters(filterSize, numberOfFilters)


    def prepareImageForConvolution(self, image):
        paddedImage = np.pad(image, self.paddingSize)
        imageTensor = np.reshape(paddedImage, (1, self.inputSize, self.inputSize))
        return np.repeat(imageTensor, self.numberOfFilters, 0)


    def convolute(self, image):
        inputTensor = self.prepareImageForConvolution(image)

        for i in range(self.inputSize + 2 * self.paddingSize):
            for j in range(self.inputSize + 2 * self.paddingSize):
                inputSlice = inputTensor[:, i:i + self.filterSize, j:j + self.filterSize]
                mult = inputSlice * self.filterTensor
                net = np.sum(mult, (1,2))

                self.featureMapTensor[:, i, j] = net

        self.featureMapTensor = self.activationFun(self.featureMapTensor)


    def pool(self):
        for i in range(self.featureMapSize)[::self.poolingWindowSize]:
            for j in range(self.featureMapSize)[::self.poolingWindowSize]:
                featureMapSlice = self.featureMapTensor[:, i:i + self.poolingWindowSize, j:j + self.poolingWindowSize]
                
                self.poolingFeatureMapTensor[:, i, j] = np.max(featureMapSlice, (1,2))


    def feedForward(self, image):
        self.convolute(image)
        self.pool()
        self.mlpNetwork.feedForward(self.poolingFeatureMapTensor.flatten())


    def backpropagateError(self, expectedOutput):
        self.mlpNetwork.backPropagateError(self.mlpNetwork.outputLayerActivations[-1], expectedOutput)
        self.

