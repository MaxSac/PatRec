import numpy as np
import NeuronalNet as NN

L2 = NN.EuclideanLoss()

Perc = NN.Perceptron(2, 2,NN.SigmoidLayer(),L2,1,1,1)
#print(Perc.forward(np.array([[7,11],[7,11]])))
#Perc.estimate(np.array([[7,11],[7,11]]), [[1,1],[1,1]])


#L2.loss(np.array([[0,0,0],[1,2,3]]),np.array([[1,2,3],[1,2,3]]))
#L2.gradient()

#print(Perc.forward(np.array([[7,11],[7,11]])))
print('Weights: ')
print(Perc.fc_lay.weights)
print('-------------------------------')
Perc.estimate(np.array([[7,11]]),np.array([[1,2]]))
