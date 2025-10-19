"""logistic regression self implemented""" 
import random
import math
import numpy as np
import utils
from matplotlib import pyplot as plt
features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
labels = np.array([0,0,0,0,1,1,1,1])


utils.plot_points(features, labels)


def logistic_trick(weights, bias, feature, label, learning_rate=0.1):
    """perform logistic trick to move weights and bias"""
    prediction = calculate_prediction(feature, weights, bias)
    # pred_res = 1 if prediction_rate > 0 else 0
    # if label == pred_res:
    # if prediction is correct, label - prediction is positive
    weights[0] += learning_rate * (label - prediction[0]) * feature[0]
    weights[1] += learning_rate * (label - prediction[0]) * feature[1]
    for i, weight in enumerate(weights):
        weight += learning_rate * (label - prediction[0]) * feature[i]
        bias += learning_rate * (label - prediction[0])
    return weights, bias, prediction[0]

def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))

def calculate_prediction(feature, weights, bias):
    """calculate the predicted rate and 0/1 result based on point + current weights and bias"""
    # Calculate the weighted sum as a single scalar using np.dot()
    weighted_sum = np.dot(weights, feature) + bias
    # print(f'weighted sum: {weighted_sum}')
    # Pass the single scalar value to the sigmoid function
    prediction_rate = sigmoid(weighted_sum)
    ##prediction_rate range (0,1), should be compared with 0.5 to get 0/1
    predict_res = 1 if prediction_rate > 0.5 else 0
    return prediction_rate, predict_res

def group_log_loss(weights, bias):
    """calculate the total log loss with set features and labels under given weights and bias"""
    errors = []
    for i,feature in enumerate(features):
        # calculate_prediction returns 2 varaiables : prediction rate and prediction result
        # therefore prediction[0] is the prediction rate,  
        # using that to avoid unused variable Pylint warning
        prediction = calculate_prediction(feature, weights, bias)
        # if predict_res == labels[i]:
        #     errors.append(-np.log(prediction_rate))
        # else:
        #     errors.append(-np.log(1 - prediction_rate))
        errors.append(log_loss(prediction[0], labels[i])) #prediction[0] is the rated prediction
    return math.prod(errors)

def log_loss(prediction, label):
    """calculate log loss for single prediction and label"""
    ## label is either 0 or 1, so only one of the two terms will be non-zero
    ## hence statement can be condensed into below formula
    return - label * np.log(prediction) - (1 - label) * np.log(1 - prediction)

def logistic_algorithm(features_v, labels_v, learning_rate=0.1, epochs=200):
    """perform logistic algorithm and set up learning rate and epochs"""
    weights = [1,2]
    bias = 0.0
    errors_t = []
    for epoch in range(epochs):
        i = random.randint(0, len(features_v)-1)
        weights, bias, pred_rate = logistic_trick(weights, bias, features_v[i], labels_v[i], learning_rate)
        error = group_log_loss(weights, bias)
        print(f'epoch {epoch}, weights: {weights}, bias: {bias}, pred_rate : {pred_rate}, total log loss: {error}')
        errors_t.append(error)
        utils.draw_line(weights[0], weights[1], bias)
    utils.plot_points(features_v, labels_v)
    plt.show()
    plt.scatter(range(epochs), errors_t)
    plt.show()

    return weights, bias, errors_t


logistic_algorithm(features, labels, learning_rate=0.1, epochs=200)
