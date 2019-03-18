#biasa disebut dengan cross validation, di hitung dari skor
#atau dapat menggunakan K-Fold cross validation
#random_state=1 digunakan agar fitting tidak random

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#membandingkan logreg, svm, dan random forest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score

digits=load_digits()
print(dir(digits))
x=digits['data']
y=digits['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1,random_state=1)

# print(len(x_train))
# print(len(x_test))

logreg=LogisticRegression(multi_class='auto',solver='liblinear')
logreg.fit(x_train,y_train)
skorlogreg=logreg.score(x_test,y_test)
# print(skorlogreg*100, ' %')

suppvm=SVC(gamma='auto')
suppvm.fit(x_train,y_train)
skorsvm=suppvm.score(x_test,y_test)
# print(skorsvm*100, ' %')

ranfor=RandomForestClassifier(n_estimators=50)
ranfor.fit(x_train,y_train)
skorranfor=ranfor.score(x_test,y_test)
# print(skorranfor*100, ' %')

print(cross_val_score(logreg,x,y,cv=5))
print(cross_val_score(suppvm,x,y,cv=5))
print(cross_val_score(ranfor,x,y,cv=5))