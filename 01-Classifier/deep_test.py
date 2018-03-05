import numpy as np
import NeuronalNet as NN
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

data = load_iris()
x_data = normalize(data['data'])
y_data = data['target']
new_label = np.zeros([len(y_data), int(max(y_data)+1)])
for x in range(len(y_data)):
    new_label[x,y_data[x]] =1 

multi_Perc = NN.MultilayerPerceptron(4,[4],3,'Sigmoid', 'Sigmoid',
        NN.EuclideanLoss(), 4,100,1,0.999)
multi_Perc.estimate(x_data, normalize(new_label))
pred = multi_Perc.forward(x_data)


for x in range(len(y_data)):
    print(np.round(pred[x],2), np.argmax(pred[x]), y_data[x])
    #print(np.round(pred[x,:]), y_data[x])
