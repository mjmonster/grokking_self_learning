import numpy as np
import matplotlib
from matplotlib import pyplot

def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    pyplot.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)

def scatter(features, labels):
    pyplot.scatter(features, labels);

def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    pyplot.scatter(X, y)
    pyplot.xlabel('number of rooms')
    pyplot.ylabel('prices')

def show_plot():
    # pyplot.colorbar(pyplot.cm.ScalarMappable(cmap='plasma'))
    pyplot.show()

def rmse(actual, pred):
    differences = np.subtract(actual, pred)
    return np.sqrt(np.mean(np.power(differences, 2)))
