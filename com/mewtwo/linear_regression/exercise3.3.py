import numpy as np

import utils

sizes = np.array([100, 200, 200, 250, 325])
prices = np.array([200, 475, 400, 520, 735])

predicted_prices = []

for size in sizes:
    predicted_prices.append(size * 2 + 50)

print('absolute error: ', np.sum(np.absolute(np.subtract(predicted_prices, prices)))/len(predicted_prices))
print('square error ', utils.rmse(prices, predicted_prices))
