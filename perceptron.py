from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

def regular_perceptron(x_train, y_train):
    perceptron = Perceptron(max_iter = 1000, random_state = 42, class_weight = "balanced")
    perceptron.fit(x_train, y_train)
    return perceptron