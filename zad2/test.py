import neural_network
import data_loader
import helpers

hiddenLayerSize = 100
inputSize = 28 * 28
outputSize = 10
minweight = -0.0001
maxweight = 0.0001
minbias = -0.001
maxbias = 0.001
batchSize = 20
learningRate = 0.01
desiredError = 0.01
maxEpoch = 100
validationSetSize = 1000
minDiff = 0.0001

inputs = data_loader.loadTrainInputs()
labels = data_loader.loadTrainOutputs()

network = neural_network.NeuralNetwork(
    hiddenLayerSize, inputSize, outputSize,
    minweight, maxweight, minbias, maxbias,
    helpers.relu
)

print('Learning...')
neural_network.teachNeuralNetworkWithInput(
    network,
    batchSize,
    learningRate,
    desiredError,
    maxEpoch,
    validationSetSize,
    minDiff,
    inputs,
    labels
)
print('Finished.')