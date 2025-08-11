from pyexpat import model
import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt
import utils

data = tc.SFrame('/home/mewtwo/py_workspace/grokking_self_learning/com/mewtwo/linear_regression/Hyderabad.csv')

print('Number of rows: ', data.num_rows())



model = tc.linear_regression.create(data, target='Price')

# house = tc.SFrame({'Area': [1000], 'No_of_Bedrooms': [3]})

# model.predict(house)

# plt.scatter(data['Area'], data['Price'])

# plt.show()

model.coefficients

model.evaluate(data)
house = tc.SFrame({'Area': [1000], 'No. of Bedrooms':[3]})
house

model.predict(house)
