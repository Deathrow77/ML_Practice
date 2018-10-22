#Itnaa optimized code?
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)
features = np.array(df.drop(['class']))
labels = np.array(df['class'])
X_train, Y_train, X_test, Y_test = test_train_split(features, labels, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
data = [5,2,1,1,2,1,1,1,2]
prediction = clf.predict(data)
print(prediction)
