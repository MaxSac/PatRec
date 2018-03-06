import numpy as np
import NeuronalNet as NN
from sklearn.preprocessing import normalize

L2 = NN.EuclideanLoss()
Sigmoid = NN.SigmoidLayer()

print('Data Info: -------------------------------')
train_data = np.array(
    [
     [0.7,0.6]
    ]
)
print('train_data:', train_data.shape)
print(train_data)
train_label = np.array(
    [
     [0.9,0.2]
    ]
)
print('train_label:', train_label.shape)
print(train_label)


# print('Perceptron Info: -------------------------------')
# Perc = NN.Perceptron(3, 2, Sigmoid,L2,0.5,10,learning_rate=0.5)
# 
# print('Gewichte des Perceptrons', Perc.fc_lay.weights.shape)
# print(Perc.fc_lay.weights)
# print('Forward des Perceptrons', Perc.forward(np.array(train_data)).shape)
# print(Perc.forward(np.array(train_data)))
# print('Training des Perceptrons')
# Perc.estimate(train_data, train_label)
# print('Forward des Perceptrons nach dem Training', 
#         Perc.forward(np.array(train_data)).shape)
# print(Perc.forward(np.array(train_data)))

print('MultilayerPerceptron Info: -------------------------------')
batch_size = 1
learning_rate = 10
momentum = 1
epochs = 1
multi_Perc = NN.MultilayerPerceptron(2,[2],
         2,Sigmoid,Sigmoid,L2,batch_size,epochs=epochs,learning_rate=learning_rate,momentum=momentum)

# print(multi_Perc.forward(train_data[:1]))
# print('-------------')
# print('Training data: ', train_data[:1])
multi_Perc.layers[1].fc_lay.weights = np.array([[0.2,0.1],[0.4, -0.4],[0.3,0.9]])
#print('weights: ', multi_Perc.layers[0].fc_lay.weights)
multi_Perc.layers[0].fc_lay.weights = np.array([[0.3,-0.2],[0.8,-0.6],[0.5,0.7]])
#print('weights: ', multi_Perc.layers[1].fc_lay.weights)

#print(multi_Perc.forward(train_data))

# print(multi_Perc.forward(normalize(train_data)))
# multi_Perc.estimate(normalize(train_data), normalize(train_label))
multi_Perc.estimate(train_data, train_label)
# print(multi_Perc.forward(normalize(train_data)))

#print(multi_Perc.forward(normalize(train_data)))
