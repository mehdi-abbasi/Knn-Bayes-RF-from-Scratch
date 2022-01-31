import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import bayes

feature, target = load_iris(return_X_y=True)
feature_train, feature_test, target_train, target_test = train_test_split(feature,target,test_size=0.3,random_state=42)


model = bayes.GaussianNB(feature_train,target_train)
pred = model.predict(feature_test)
print((pred == target_test).sum()/len(pred) * 100)

knn = bayes.Knn(feature_train,target_train)
knn_pred = knn.query(feature_test,k=3)
print((knn_pred == target_test).sum()/len(knn_pred) * 100)
