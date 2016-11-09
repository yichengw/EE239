import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

with open('D:\\study\\Winter 2016\\Big Data\\Project 1\\housing_data.csv', newline = '') as f :
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

final_rmse = 0

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

reg = LinearRegression()
clf = Lasso(alpha=0.001)

for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] :
    i = j*50
    mse = 0
    y_cp = my_list_MEDV[:]
    x_CRIM_cp = my_list_CRIM[:]
    x_ZN_cp = my_list_ZN[:]
    x_INDUS_cp = my_list_INDUS[:]
    x_CHAS_cp = my_list_CHAS[:]
    x_NOX_cp = my_list_NOX[:]
    x_RM_cp = my_list_RM[:]
    x_AGE_cp = my_list_AGE[:]
    x_DIS_cp = my_list_DIS[:]
    x_RAD_cp = my_list_RAD[:]
    x_TAX_cp = my_list_TAX[:]
    x_PTRATIO_cp = my_list_PTRATIO[:]
    x_B_cp = my_list_B[:]
    x_LSTAT_cp = my_list_LSTAT[:]

    del y_cp[j*50:(j+1)*50]
    del x_CRIM_cp[j*50:(j+1)*50]
    del x_ZN_cp[j*50:(j+1)*50]
    del x_INDUS_cp[j*50:(j+1)*50]
    del x_CHAS_cp[j*50:(j+1)*50]
    del x_NOX_cp[j*50:(j+1)*50]
    del x_RM_cp[j*50:(j+1)*50]
    del x_AGE_cp[j*50:(j+1)*50]
    del x_DIS_cp[j*50:(j+1)*50]
    del x_RAD_cp[j*50:(j+1)*50]
    del x_TAX_cp[j*50:(j+1)*50]
    del x_PTRATIO_cp[j*50:(j+1)*50]
    del x_B_cp[j*50:(j+1)*50]
    del x_LSTAT_cp[j*50:(j+1)*50]
    y1 = y_cp
    x1 = [x_CRIM_cp, x_ZN_cp, x_INDUS_cp, x_CHAS_cp, x_NOX_cp, x_RM_cp, x_AGE_cp, x_DIS_cp, x_RAD_cp, x_TAX_cp, x_PTRATIO_cp, x_B_cp, x_LSTAT_cp]

    y1_a = np.asarray(y1)
    x1_a = np.asarray(x1).T

    poly = PolynomialFeatures(3)
    x1_p = poly.fit_transform(x1_a)
    y1_p = y1_a

    clf.fit(x1_p, y1_p)
    
    while i < 50*(j+1) :
        x_t = np.asarray([my_list_CRIM[i], my_list_ZN[i], my_list_INDUS[i], my_list_CHAS[i], my_list_NOX[i], my_list_RM[i], my_list_AGE[i], my_list_DIS[i], my_list_RAD[i], my_list_TAX[i], my_list_PTRATIO[i], my_list_B[i], my_list_LSTAT[i]]).reshape(1, -1)
        x_t_p = poly.fit_transform(x_t)
        y_prediction = clf.predict(x_t_p)
        y_prediction_list = y_prediction.tolist()[0]
        mse = mse + (y_prediction_list - my_list_MEDV[i])**2
        i = i + 1


    final_rmse = final_rmse + math.sqrt(mse/50)
    print(math.sqrt(mse/50))

print('average rmse is:')
print(final_rmse/10)
