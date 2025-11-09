"""Decision Tree Classifier for University Admission Prediction"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from com.mewtwo.decision_tree.utils import save_tree

data = pd.read_csv("admission_predict.csv")
data['Admitted'] = data['Chance of Admit'] >= 0.75
data = data.drop(['Chance of Admit'], axis=1)

features = data.drop('Admitted', axis=1)
labels = data['Admitted']

# dt = DecisionTreeClassifier()

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, min_samples_split=10)
dt.fit(features, labels)

print(dt.predict(features[0:5]))

save_tree(dt)
