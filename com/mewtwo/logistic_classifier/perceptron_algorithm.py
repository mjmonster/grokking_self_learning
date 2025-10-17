from matplotlib import pyplot as plt
import numpy as np
import utils
import random


learning_rate = 0.01
epochs = 1000

def step(num):
    return 1 if num >= 0 else 0

def score(weights, bias, features):
    return np.dot(features, weights) + bias

def prediction(weights, bias, features):
    return step(score(weights, bias, features))

xn = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
y = np.array([0,0,0,0,1,1,1,1])

utils.plot_points(xn, y)
utils.draw_line(1, -1, 0, starting=0, ending=3, color='black')


def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(score(weights, bias, features))
    

def mean_perceptron_error(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])
    return total_error / len(features)

def perceptron_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label - pred) * features[i] * learning_rate
    bias += (label - pred) * learning_rate
    return weights, bias

random.seed(0)


def perceptron_algorithm(features, labels, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for epoch in range(epochs):
        error = mean_perceptron_error(weights, bias, features, labels)
        errors.append(error)
        i = random.randint(0, len(features)-1)
        weights, bias = perceptron_trick(weights, bias, features[i], labels[i], learning_rate)
        utils.draw_line(weights[0], weights[1], bias)
    
    utils.plot_points(features, labels)
    plt.show()
    plt.scatter(range(epochs), errors)
    return weights, bias, errors

perceptron_algorithm(xn, y)
# utils.show_plot()
