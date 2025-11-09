"""Decision Tree Classifier for University Admission Prediction"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from com.mewtwo.decision_tree.utils import save_tree

data = pd.read_csv("admission_predict.csv")
data['Admitted'] = data['Chance of Admit'] >= 0.75
data = data.drop(['Chance of Admit'], axis=1)

features = data.drop('Admitted', axis=1)
labels = data['Admitted']

dt = DecisionTreeClassifier()
dt.fit(features, labels)
print(dt.predict(features[0:5]))

dt_restricted = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, min_samples_split=10)
dt_restricted.fit(features, labels)
#                             337,118,4,4.5,4.5,9.65,1,0.92
print(dt_restricted.predict([[500,320,110,3,4.0,3.5,8.9,0]]))

save_tree(dt)
