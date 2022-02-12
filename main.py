import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import bayes

feature, target = load_iris(return_X_y=True)
feature_train, feature_test, target_train, target_test = train_test_split(feature,target,test_size=0.3,random_state=42)


# model = bayes.GaussianNB(feature_train,target_train)
# pred = model.predict(feature_test)
# print((pred == target_test).sum()/len(pred) * 100)

# knn = bayes.Knn(feature_train,target_train)
# knn_pred = knn.query(feature_test,k=3)
# print((knn_pred == target_test).sum()/len(knn_pred) * 100)

# rn = bayes.Tree(feature_train,target_train,4)

# tt = rn.create_tree()
# print(tt[1].feature)
# print(tt[1].threshold)
# print(tt[1].dataset)

# print(tt[0].dataset,tt[1].dataset,tt[2].dataset)

# print(len(tt[0].dataset))
# print(len(tt[1].dataset),len(tt[2].dataset))
# print(len(tt[3].dataset),len(tt[4].dataset),len(tt[5].dataset),len(tt[6].dataset))

# print(tt[0].group,tt[0].threshold)
# print(tt[1].group,tt[1].threshold,tt[2].group,tt[2].threshold)
# print(tt[3].group,tt[3].threshold,tt[4].group,tt[4].threshold,tt[5].group,tt[5].threshold,tt[6].group,tt[6].threshold)
# print(tt[3].dataset,tt[4].dataset,tt[5].dataset,tt[6].dataset)
# print(tt[3].group,tt[4].group,tt[5].group,tt[6].group)


# print(rn.create_tree())

rf = bayes.RandomForest(feature_train,target_train,4,2)
rf.build()
pp = rf.predict(feature_test)

print(pp)


