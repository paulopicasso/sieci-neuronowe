import numpy as np
import idx2numpy

def loadTrainInputs(parentDir = '') -> np.ndarray:
    return loadInputs(parentDir + 'assets/train-images.idx3-ubyte')


def loadTrainOutputs(parentDir = '') -> np.ndarray:
    return loadOutputs(parentDir + 'assets/train-labels.idx1-ubyte')


def loadInputs(path: str) -> np.ndarray:
    images = idx2numpy.convert_from_file(path)
    return images


def loadOutputs(path: str) -> np.ndarray:
    labels = idx2numpy.convert_from_file(path)
    length = np.size(labels, 0)
    zeros = np.zeros((length, 10))
    for i in range(length):
        label = labels[i]
        zeros[i][label] = 1
    
    return zeros
