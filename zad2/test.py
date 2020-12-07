import neural_network
import data_loader
import helpers

hiddenLayerSize = 200
inputSize = 28 * 28
outputSize = 10
minweight = -0.001
maxweight = 0.001
minbias = -0.01
maxbias = 0.01
batchSize = 20
learningRate = 0.001
desiredError = 0.01
maxEpoch = 100
validationSetSize = 10000
minDiff = 0.0001

inputs = data_loader.loadTrainInputs()
labels = data_loader.loadTrainOutputs()

network = neural_network.NeuralNetwork(
    hiddenLayerSize, inputSize, outputSize,
    minweight, maxweight, minbias, maxbias,
    helpers.relu,
    # useMomentum=True,
    # useNag=True,
    # momentumRate=0.8
    # weightInitializer=helpers.heInitializeWeights
    learningRateMode='adam'
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
