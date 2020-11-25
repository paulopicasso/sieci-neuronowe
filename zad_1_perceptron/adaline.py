import perceptron
import numpy as np

def learnAdaline(data, alpha, minWeight, maxWeight, targetError):
    weights = perceptron.getWeights(3, minWeight, maxWeight)
    inputs = perceptron.getInputFromData(data)
    expectedOutputs = perceptron.getOutputFromData(data)
    # print('Weights: ', weights)

    epoch = 1
    error = targetError

    while error >= targetError:
        errors = np.array([])

        for i, inputRow in enumerate(inputs):
            activationSum = np.sum(inputRow * weights)
            delta = expectedOutputs[i] - activationSum
            errors = np.append(errors, delta * delta)

            weights = [w + (alpha * delta * inputRow[k]) for k, w in enumerate(weights)]


        error = np.sum(errors) / np.size(errors)
        # print('Epoch: ', epoch)
        # print('Error: ', error)
        # print('Weights: ', weights)
        epoch += 1

    return (epoch, weights, error)

# ranges = [
#     (-1, 1),
#     (-0.8, 0.8),
#     (-0.6, 0.6),
#     (-0.4, 0.4),
#     (-0.2, 0.2),
# ]
# alphas = [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07]

# import perceptron_const

# for ws in ranges:
#     (wmin, wmax) = ws
#     print('\nWmin: ', wmin, ' Wmax: ', wmax)
#     for a in alphas:
#         epochs = np.array([])
#         for i in range(10):
#             (epoch, _, _) = learnAdaline(perceptron_const.learnDataBipolar, a, wmin, wmax, 0.3)
#             epochs = np.append(epochs, epoch)

#         averageEpochs = np.sum(epochs) / np.size(epochs)
#         print(averageEpochs, end=' ')
