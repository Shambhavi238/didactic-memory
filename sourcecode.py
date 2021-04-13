# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:30:36 2020

@author: ankit
"""

'''import numpy as np'''
import pandas as pd
'''import matplotlib.pyplot as plt'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
data=pd.read_csv('diabetes.csv')
print(data.head())
real_x = data.iloc[:,[0,1,2,3,4,5,6,7]].values
real_y = data.iloc[:,8].values
print(real_x)
print(real_y)

training_x, test_x, training_y, test_y=train_test_split(real_x,real_y,test_size=0.25,random_state=0)
ss=StandardScaler()
training_x = ss.fit_transform(training_x)
test_x = ss.fit_transform(test_x)
print(training_x)

cls_LR = LogisticRegression(random_state=0)
cls_LR.fit(training_x,training_y)
y_pred = cls_LR.predict(test_x)

print('-------------------original data-------------------')
print(test_y)

print('-------------------prediction----------------------')
print(y_pred)

c_m = confusion_matrix(test_y, y_pred)
print(c_m)

print("Accuracy : {0:.4f}".format(metrics.accuracy_score(test_y,y_pred)))

'''plt.plot()
plt.show()'''