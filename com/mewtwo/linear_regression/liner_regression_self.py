from com.mewtwo.linear_regression import utils
from matplotlib import pyplot as plt
import numpy as np

'''
Liner Regression -
y=slope * x + intercept
'''
import random

colors = plt.cm.viridis(np.linspace(0, 1, 5))

# slope = 1
# intercept = 1
# rate = 1

b = 0
slope = 0.8
intercept = 0.5
s_rate = 1
i_rate = 1
# arr_xy = [[1,5], [2,7],[3,9],[4,11],[5,13],[6,15],[7,17]]
arr_xy = [[1,155], [2,197],[3,244],[5,356],[6,407],[7,448]]
arr_x = []
arr_y = []
for i in arr_xy:
    arr_x.append(i[0])
for i in arr_xy:
    arr_y.append(i[1])

utils.plot_points(arr_x, arr_y)


def calculate(x, y, color):
    ## when x = input x, what is the y value?
    ## y = slope * x + intercept
    ##calculate slope direction, if equation turns close to the point, move forward, otherwise move backward
    ##calculate when x equals input in the formula, what is the y value
    global slope, intercept, b
    utils.draw_line(slope, intercept, color=color, starting=0, ending=8)
    predicted_price = slope * x + intercept
    # x_on_line = (y - intercept) / slope
    predicted_price_plus_rate = predicted_price + i_rate
    # if x_on_line > x:
    #     slope = slope+s_rate
    # elif x_on_line < x:
    #     slope = slope-s_rate

    if predicted_price > y and x > 0:
        intercept -= i_rate
        slope -= s_rate
    elif predicted_price > y and x < 0:
        intercept += i_rate
        slope -= s_rate
    elif predicted_price < y and x > 0 :
        intercept += i_rate
        slope += s_rate
    elif predicted_price < y and x < 0:
        intercept -= i_rate
        slope += s_rate
    print("y=", slope , '*x', '+', intercept)


cmap = plt.get_cmap('YlOrRd')
n_lines = 300
colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

while int(b)<n_lines:
    temp_list = random.choice(arr_xy)
    calculate(temp_list[0], temp_list[1], colors[b])
    b+=1


utils.show_plot()
