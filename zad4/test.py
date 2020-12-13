import network as cnn

network = cnn.Network()
network.learn(0.1, 0.01, 0.01, 100, 0.00001, validationSetSize=1000)