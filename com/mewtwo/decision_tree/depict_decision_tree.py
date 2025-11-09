"""Class to depict a decision tree using sklearn"""
import pandas as pd
import utils

from sklearn.tree import DecisionTreeClassifier
from com.mewtwo.decision_tree.utils import save_tree

dataset = pd.DataFrame({
    'x_0':[7,3,2,1,2,4,1,8,6,7,8,9],
    'x_1':[1,2,3,5,6,7,9,10,5,8,4,6],
    'y':[0,0,0,0,0,1,1,1,1,1,1,1]
})

features = dataset[['x_0','x_1']]
labels = dataset['y']

utils.plot_points(features, labels)
utils.show_plot()
decision_tree = DecisionTreeClassifier(criterion="gini")
decision_tree.fit(features, labels)

save_tree(decision_tree)
