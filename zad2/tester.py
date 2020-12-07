import neural_network
import data_loader
import helpers
import numpy as np

inputSize = 28 * 28
outputSize = 10
minWeight = -0.1
maxWeight = 0.1
batchSize = 20
desiredError = 0.01
maxEpoch = 100
minDiff = 0.00001
layerSize = 200

def test(
    experimentName,
    learningRate=0.1,
    useMomentum=False,
    momentumRate=0.8,
    useNag=False,
    weightInitializer=helpers.normalInitializeWeights,
    learningRateMode='standard',
    activationFun=helpers.relu
):
    inputs = data_loader.loadTrainInputs('../')
    labels = data_loader.loadTrainOutputs('../')

    avgEpochs = []
    avgAcc = []
    for i in range(10):
        network = neural_network.NeuralNetwork(
            layerSize, 
            inputSize, 
            outputSize, 
            minWeight,
            maxWeight, 
            minWeight, 
            maxWeight,
            activationFun,
            useMomentum=useMomentum,
            momentumRate=momentumRate,
            useNag=useNag,
            weightInitializer=weightInitializer,
            learningRateMode=learningRateMode
        )
        (epoch, accuracies) = neural_network.teachNeuralNetworkWithInput(
            network, 
            batchSize, 
            learningRate, 
            desiredError, 
            maxEpoch, 
            10000, 
            minDiff, 
            inputs,
            labels, 
        )
        print('i: ', i)
        avgAcc.append(accuracies)
        avgEpochs.append(epoch)

    epoch = np.average(avgEpochs)
    accuracy = np.average(avgAcc, 0)

    with open(experimentName + '.txt', 'a') as fp:
        fp.write('{}'.format(epoch) + '\n')
        fp.write('{}'.format(accuracy) + '\n')

