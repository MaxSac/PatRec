import numpy as np
import NeuronalNet as NN
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
x_data = normalize(data['data'])
y_data = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3)

new_label = np.zeros([len(y_train), int(max(y_train)+1)])
for x in range(len(y_train)):
    new_label[x,y_train[x]] = 1 

multi_Perc = NN.MultilayerPerceptron(4,[10,10],3,'Sigmoid', 'Sigmoid',
        NN.EuclideanLoss(),64, 10000, 0.1, 0.9)
multi_Perc.estimate(X_train, normalize(new_label))
pred = multi_Perc.forward(X_train)
Pred = multi_Perc.forward(X_test)

clf = RandomForestClassifier(max_depth=20, n_estimators=20, n_jobs=-1)
clf.fit(X_train, y_train)
print('Accuracy RF {:.2}'.format(np.mean(y_test==clf.predict(X_test))))

truth = np.empty(0)
Truth = np.empty(0)


for x in range(len(y_train)):
    truth = np.append(truth, np.argmax(pred[x]))
print('Accuracy training: {:.2}'.format(np.mean(truth == y_train)))

for x in range(len(y_test)):
    # print(np.round(pred[x],2), np.argmax(pred[x]), y_train[x])
    Truth = np.append(Truth, np.argmax(Pred[x]))

print('Accuracy test: {:.2}'.format(np.mean(Truth == y_test)))
