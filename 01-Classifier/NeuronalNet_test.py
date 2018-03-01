from __future__ import print_function
import numpy as np
import NeuronalNet as NN

L2 = NN.EuclideanLoss()
Sigmoid = NN.SigmoidLayer()

print('\nData Info:')
print('----------')
train_data = np.array(
    [
     [1,  2, 3],
     [1,  2, 3],
     [50, 2, 3],
     [1,  2, 2],
     [1,  2, 1]
    ]
)
print('train_data:', train_data.shape)
print(train_data)
train_label = np.array(
    [
     [1, 1],
     [1, 1],
     [0, 1],
     [1, 0],
     [1, 0]
    ]
)
print('train_label:', train_label.shape)
print(train_label)


def pereptron():
    print('\nPerceptron Info:')
    print('----------------')
    Perc = NN.Perceptron(
        n_input=3,
        n_output=2,
        activation_function=Sigmoid,
        loss=L2,
        batch_size=0.5,
        epochs=10,
        learning_rate=0.5,
    )

    print('Gewichte des Perceptrons', Perc.fc_lay.weights.shape)
    print(Perc.fc_lay.weights)
    print('Forward des Perceptrons', Perc.forward(np.array(train_data)).shape)
    print(Perc.forward(np.array(train_data)))
    print('Training des Perceptrons')
    Perc.estimate(train_data, train_label)
    print('Forward des Perceptrons nach dem Training',
            Perc.forward(np.array(train_data)).shape)
    print(Perc.forward(np.array(train_data)))

print('\nMultilayerPerceptron Info:')
print('--------------------------')
multi_Perc = NN.MultilayerPerceptron(
    n_input=3,
    n_hidden=[3],
    n_output=2,
    activation_function_hidden=NN.SigmoidLayer(),
    activation_function_output=NN.SigmoidLayer(),
    loss=L2,
    batch_size=1,
    epochs=1,
    learning_rate=1,
    momentum=1,
)
print(multi_Perc.forward(train_data))
print('-------------')
multi_Perc.estimate(train_data, train_label)
# #multi_Perc.estimate(np.array([[7,11,1,2], [7,11,1,2]]),np.array([[1], [1]]))
