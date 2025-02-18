# Our import statements for this problem
import pandas as pd
import numpy as np
import patsy as pt


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

test_data = pd.read_csv("assignment2test.csv")
train_data = pd.read_csv("assignment2train.csv")
y = train_data['meal']
x = train_data.drop(['meal','id','DateTime'], axis = 1)

model = DecisionTreeClassifier(max_depth=100, min_samples_leaf=10)

modelFit = model.fit(x,y)


xt = test_data.drop(['meal', 'id', 'DateTime'], axis=1)

pred = modelFit.predict(xt)
