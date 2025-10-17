from email import errors
from math import log
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
import random
import utils

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
labels = np.array([0,0,0,0,1,1,1,1])


utils.plot_points(features, labels)

def logistic_trick(weights, bias, feature, label, learning_rate=0.1):

    prediction_rate = 1 / (1 + np.exp(-(weights[0]*feature[0] + weights[1]*feature[1] + bias)))
    pred_res = 1 if prediction_rate > 0.5 else 0
    # if label == pred_res:
    # if prediction is correct, label - prediction is positive
    weights[0] += learning_rate * (label - prediction_rate) * feature[0]
    weights[1] += learning_rate * (label - prediction_rate) * feature[1]
    bias += learning_rate * (label - prediction_rate)
    return weights, bias, prediction_rate


def log_loss(label, prediction_rate):
    if label == 1:
        return -np.log(prediction_rate)
    else:
        return -np.log(1 - prediction_rate)

def logistic_algorithm(features, labels, learning_rate=0.1, epochs=200):
    weights = [1,2]
    bias = 0.0
    errors = []
    for epoch in range(epochs):
        i = random.randint(0, len(features)-1)
        weights, bias, pred_rate = logistic_trick(weights, bias, features[i], labels[i], learning_rate)
        error = log_loss(labels[i], pred_rate)
        print(f'epoch {epoch}, weights: {weights}, bias: {bias}, error: {error}')
        errors.append(error)
        utils.draw_line(weights[0], weights[1], bias)
    utils.plot_points(features, labels)
    plt.show()
    # plt.scatter(range(epochs), errors)
    return weights, bias, errors


logistic_algorithm(features, labels, learning_rate=0.1, epochs=200)

