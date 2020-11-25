
import numpy as np

def learnSimple (data, alpha, minWeight, maxWeight, activationFuncName):
    activationFunction = unipolar

    if activationFuncName == 'bipolar':
        data = np.array([convertToBipolar(row) for row in data])
        activationFunction = bipolar

    weights = getWeights(3, minWeight, maxWeight)
    inputs = getInputFromData(data)
    expectedOutputs = getOutputFromData(data)

    # print('Initial weights: ', weights)

    epoch = 1
    errorInEpoch = True
    while errorInEpoch:
        
        errorInEpoch = False
        for i, inputRow in enumerate(inputs):
            activationSum = np.sum(inputRow * weights)
            activationValue = activationFunction(activationSum)
            error = expectedOutputs[i] - activationValue

            if error != 0:
                errorInEpoch = True

            weights = [w + (alpha * error * inputRow[k]) for k, w in enumerate(weights)]

        # print('Epoch: ', epoch)
        # print('Weights: ', weights)

        epoch += 1
    
    return (epoch, weights)

def getWeights(number, min, max):
    rand = np.random.rand(number)
    return rand * (max - min) + min

def getInputFromData(data):
    baseInputs = [row[0:2] for row in data]
    return np.array([np.concatenate(([1], row)) for row in baseInputs])

def getOutputFromData(data):
    return [row[2] for row in data]

def unipolar(z):
    return 1 if z > 0 else 0

def bipolar(z):
    return 1 if z > 0 else -1

def convertToBipolar(list):
    return [-1 if x == 0 else x for x in list]
        
# ranges = [
#     (-1, 1),
#     (-0.8, 0.8),
#     (-0.6, 0.6),
#     (-0.4, 0.4),
#     (-0.2, 0.2),
# ]
# alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0, 2.0]

# import perceptron_const

# for ws in ranges:
#     (wmin, wmax) = ws
#     print('\nWmin: ', wmin, ' Wmax: ', wmax)
#     for a in alphas:
#         epochs = np.array([])
#         for i in range(10):
#             (epoch, _) = learnSimple(perceptron_const.learnDataBipolar, a, wmin, wmax, 'bipolar')
#             epochs = np.append(epochs, epoch)

#         averageEpochs = np.sum(epochs) / np.size(epochs)
#         print(averageEpochs, end=' ')