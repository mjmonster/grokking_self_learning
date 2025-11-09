"""A simple example of Decision Tree Regression using sklearn."""
import com.mewtwo.decision_tree.utils as utils
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

features = [[10],[20],[30],[40],[50],[60],[70],[80]]
labels = [7,5,7,1,2,1,5,4]

dt_regressor = DecisionTreeRegressor(max_depth=2)
dt_regressor.fit(features, labels)

plt.scatter(features, labels)
plt.xlabel("Age")
plt.ylabel("Days per week")
utils.plot_regressor(dt_regressor, features, labels)

plt.show()
