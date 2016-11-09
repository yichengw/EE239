import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import cross_validation

with open('C:\\Users\\Gaoxiang\\Desktop\\EE239\\housing_data.csv', newline = '') as f :
     r = list(csv.reader(f))
my_data=r
random.shuffle(my_data)

my_list_CRIM=list()
my_list_ZN=list()
my_list_INDUS=list()
my_list_CHAS=list()
my_list_NOX=list()
my_list_RM=list()
my_list_AGE=list()
my_list_DIS=list()
my_list_RAD=list()
my_list_TAX=list()
my_list_PTRATIO=list()
my_list_B=list()
my_list_LSTAT=list()
my_list_MEDV=list()
i=0
while i<=len(my_data)-1:
    my_list_CRIM.append(float(my_data[i][0]))
    my_list_ZN.append(float(my_data[i][1]))
    my_list_INDUS.append(float(my_data[i][2]))
    my_list_CHAS.append(float(my_data[i][3]))
    my_list_NOX.append(float(my_data[i][4]))
    my_list_RM.append(float(my_data[i][5]))
    my_list_AGE.append(float(my_data[i][6]))
    my_list_DIS.append(float(my_data[i][7]))
    my_list_RAD.append(float(my_data[i][8]))
    my_list_TAX.append(float(my_data[i][9]))
    my_list_PTRATIO.append(float(my_data[i][10]))
    my_list_B.append(float(my_data[i][11]))
    my_list_LSTAT.append(float(my_data[i][12]))
    my_list_MEDV.append(float(my_data[i][13]))
    i=i+1


y1 = my_list_MEDV
x1 = np.asarray([my_list_CRIM, my_list_ZN, my_list_INDUS, my_list_CHAS, my_list_NOX, my_list_RM, my_list_AGE, my_list_DIS, my_list_RAD, my_list_TAX, my_list_PTRATIO, my_list_B, my_list_LSTAT]).T
reg = LinearRegression()
scores=cross_validation.cross_val_score(reg,x1,y1,cv=10, scoring='mean_squared_error')
rmse=math.sqrt(-np.mean(scores))

print (rmse)

