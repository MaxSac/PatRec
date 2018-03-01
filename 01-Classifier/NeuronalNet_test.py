import numpy as np
import NeuronalNet as NN

L2 = NN.EuclideanLoss()
Sigmoid = NN.SigmoidLayer()

print('Data Info: -------------------------------')
train_data = np.array([[1,2,3],[1,2,3],[50,2,3],[1,2,2],[1,2,1]])
print('train_data:', train_data.shape)
print(train_data)
train_label = np.array([[1,1],[1,1],[0,1],[1,0],[1,0]])
print('train_label:', train_label.shape)
print(train_label)


print('Perceptron Info: -------------------------------')
Perc = NN.Perceptron(3, 2, Sigmoid,L2,0.5,10,learning_rate=0.5)

print('Gewichte des Perceptrons', Perc.fc_lay.weights.shape)
print(Perc.fc_lay.weights)
print('Forward des Perceptrons', Perc.forward(np.array(train_data)).shape)
print(Perc.forward(np.array(train_data)))
print('Training des Perceptrons')
Perc.estimate(train_data, train_label)
print('Forward des Perceptrons nach dem Training', 
        Perc.forward(np.array(train_data)).shape)
print(Perc.forward(np.array(train_data)))

print('Perceptron Info: -------------------------------')
multi_Perc = NN.MultilayerPerceptron(3,[3],
         2,NN.SigmoidLayer(),NN.SigmoidLayer(),L2,1,1,1,1)
print(multi_Perc.forward(train_data))
print('-------------')
multi_Perc.estimate(train_data, train_label)
# #multi_Perc.estimate(np.array([[7,11,1,2], [7,11,1,2]]),np.array([[1], [1]]))
