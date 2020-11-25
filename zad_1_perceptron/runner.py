import perceptron
import adaline
import perceptron_const
import numpy as np

while(True):
    algorithm = int(input('Typ algorytmu (1 = perceptron prosty, 2 = Adaline):'))
    minWeight = float(input('Dolny zakres wag:'))
    maxWeight = float(input('Górny zakres wag:'))
    alpha = float(input('Współczynnik uczenia:'))
    targetError = float(input('Błąd oczekiwany (dla Adaline):'))
    activationFunc = int(input('Funkcja aktywacji (dla perceptronu prostego)(1 = unipolarna, 2 = bipolarna)'))
    runCount = int(input('Ile razy uruchomić:'))

    if algorithm == 1:
        data = perceptron_const.learnData if activationFunc == 1 else perceptron_const.learnDataBipolar
        activationFunc = 'unipolar' if activationFunc == 1 else 'bipolar'
        epochs = np.array([])

        for run in range(runCount):
            (epoch, weights) = perceptron.learnSimple(data, alpha, minWeight, maxWeight, activationFunc)
            epochs = np.append(epochs, epoch)

        averageEpochs = np.sum(epochs) / np.size(epochs)
        print('Srednia liczba epok: ', averageEpochs)

    if algorithm == 2:
        data = perceptron_const.learnDataBipolar
        epochs = np.array([])

        for run in range(runCount):
            (epoch, weights, error) = adaline.learnAdaline(data, alpha, minWeight, maxWeight, targetError)
            np.append(epochs, epoch)

        averageEpochs = np.sum(epochs) / np.size(epochs)
        print('Srednia liczba epok: ', averageEpochs)
