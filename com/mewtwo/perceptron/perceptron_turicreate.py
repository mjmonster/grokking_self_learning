from sklearn.linear_model import Perceptron
import numpy as np
import utils

# Create a Perceptron model
perceptron = Perceptron()

features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
labels = np.array([0,0,0,0,1,1,1,1])
# Fit the model to the data
perceptron.fit(features, labels)

# Predict the labels for the features
predictions = perceptron.predict(features)

print("Predictions:", predictions)

# The coefficients of the model
coefficients = perceptron.coef_[0]
intercept = perceptron.intercept_[0]

print("Coefficients:", coefficients)
print("Intercept:", intercept)

utils.plot_boundary(features, labels, coefficients, intercept)
