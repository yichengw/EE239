import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


with open('D:\\study\\Winter 2016\\Big Data\\Project 1\\housing_data.csv', newline = '') as f :
     r = list(csv.reader(f))
my_data=r
my_data_1 = r[:]
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

my_list_CRIM_1=list()
my_list_ZN_1=list()
my_list_INDUS_1=list()
my_list_CHAS_1=list()
my_list_NOX_1=list()
my_list_RM_1=list()
my_list_AGE_1=list()
my_list_DIS_1=list()
my_list_RAD_1=list()
my_list_TAX_1=list()
my_list_PTRATIO_1=list()
my_list_B_1=list()
my_list_LSTAT_1=list()
my_list_MEDV_1=list()

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

    my_list_CRIM_1.append(float(my_data_1[i][0]))
    my_list_ZN_1.append(float(my_data_1[i][1]))
    my_list_INDUS_1.append(float(my_data_1[i][2]))
    my_list_CHAS_1.append(float(my_data_1[i][3]))
    my_list_NOX_1.append(float(my_data_1[i][4]))
    my_list_RM_1.append(float(my_data_1[i][5]))
    my_list_AGE_1.append(float(my_data_1[i][6]))
    my_list_DIS_1.append(float(my_data_1[i][7]))
    my_list_RAD_1.append(float(my_data_1[i][8]))
    my_list_TAX_1.append(float(my_data_1[i][9]))
    my_list_PTRATIO_1.append(float(my_data_1[i][10]))
    my_list_B_1.append(float(my_data_1[i][11]))
    my_list_LSTAT_1.append(float(my_data_1[i][12]))
    my_list_MEDV_1.append(float(my_data_1[i][13]))
    
    i=i+1

reg = LinearRegression()
final_rmse = 0
y_prediction_for_plot_list = list()
residuals_for_plot_list = list()

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

    poly = PolynomialFeatures(2)
    x1_p = poly.fit_transform(x1_a)
    y1_p = y1_a

    reg.fit(x1_p, y1_p)
    #print('coefficients are:')
    #print(reg.coef_)
    
    while i < 50*(j+1) :
        x_t = np.asarray([my_list_CRIM[i], my_list_ZN[i], my_list_INDUS[i], my_list_CHAS[i], my_list_NOX[i], my_list_RM[i], my_list_AGE[i], my_list_DIS[i], my_list_RAD[i], my_list_TAX[i], my_list_PTRATIO[i], my_list_B[i], my_list_LSTAT[i]]).reshape(1, -1)
        x_t_p = poly.fit_transform(x_t)
        y_prediction = reg.predict(x_t_p)
        y_prediction_list = y_prediction.tolist()[0]
        if j == 9 :
            y_prediction_for_plot_list.append(y_prediction_list)
            residuals_for_plot_list.append(y_prediction_list - my_list_MEDV[i])
            
        mse = mse + (y_prediction_list - my_list_MEDV[i])**2
        i = i + 1

    print('rmse is:')
    print(math.sqrt(mse/50))

    final_rmse = final_rmse + math.sqrt(mse/50);

print('average rmse:')
print(final_rmse/10)

mse_1 = 0
y_cp_1 = my_list_MEDV_1[:]
x_CRIM_cp_1 = my_list_CRIM_1[:]
x_ZN_cp_1 = my_list_ZN_1[:]
x_INDUS_cp_1 = my_list_INDUS_1[:]
x_CHAS_cp_1 = my_list_CHAS_1[:]
x_NOX_cp_1 = my_list_NOX_1[:]
x_RM_cp_1 = my_list_RM_1[:]
x_AGE_cp_1 = my_list_AGE_1[:]
x_DIS_cp_1 = my_list_DIS_1[:]
x_RAD_cp_1 = my_list_RAD_1[:]
x_TAX_cp_1 = my_list_TAX_1[:]
x_PTRATIO_cp_1 = my_list_PTRATIO_1[:]
x_B_cp_1 = my_list_B_1[:]
x_LSTAT_cp_1 = my_list_LSTAT_1[:]

del y_cp_1[0:50]
del x_CRIM_cp_1[0:50]
del x_ZN_cp_1[0:50]
del x_INDUS_cp_1[0:50]
del x_CHAS_cp_1[0:50]
del x_NOX_cp_1[0:50]
del x_RM_cp_1[0:50]
del x_AGE_cp_1[0:50]
del x_DIS_cp_1[0:50]
del x_RAD_cp_1[0:50]
del x_TAX_cp_1[0:50]
del x_PTRATIO_cp_1[0:50]
del x_B_cp_1[0:50]
del x_LSTAT_cp_1[0:50]

y1_1 = y_cp_1
x1_1 = [x_CRIM_cp_1, x_ZN_cp_1, x_INDUS_cp_1, x_CHAS_cp_1, x_NOX_cp_1, x_RM_cp_1, x_AGE_cp_1, x_DIS_cp_1, x_RAD_cp_1, x_TAX_cp_1, x_PTRATIO_cp_1, x_B_cp_1, x_LSTAT_cp_1]

y1_a_1 = np.asarray(y1_1)
x1_a_1 = np.asarray(x1_1).T

x1_p_1 = poly.fit_transform(x1_a_1)
y1_p_1 = y1_a_1

reg.fit(x1_p_1, y1_p_1)

i = 0
while i < 50 :
    x_t_1 = np.asarray([my_list_CRIM_1[i], my_list_ZN_1[i], my_list_INDUS_1[i], my_list_CHAS_1[i], my_list_NOX_1[i], my_list_RM_1[i], my_list_AGE_1[i], my_list_DIS_1[i], my_list_RAD_1[i], my_list_TAX_1[i], my_list_PTRATIO_1[i], my_list_B_1[i], my_list_LSTAT_1[i]]).reshape(1, -1)
    x_t_p_1 = poly.fit_transform(x_t_1)
    y_prediction_1 = reg.predict(x_t_p_1)
    y_prediction_list_1 = y_prediction_1.tolist()[0]
    mse_1 = mse_1 + (y_prediction_list_1 - my_list_MEDV_1[i])**2
    i = i + 1

print('rmse for fixed training:')
print(math.sqrt(mse_1/50))
'''
plt.figure()
plt.scatter(range(50), my_list_MEDV[450:500], marker = 'x', color = 'y')
plt.hold(True)
plt.scatter(range(50), y_prediction_for_plot_list, marker = '+', color = 'b')
plt.title('fitted values and actual values scattered plot over time')
plt.show()

plt.figure()
plt.scatter(y_prediction_for_plot_list, my_list_MEDV[450:500])
plt.hold(True)
plt.plot([0, 49], [0, 49], 'r-', linewidth = 2.0)
plt.axis([0, 50, 0, 52])
plt.title('Fitted Values vs Actual Values')
plt.show()

plt.figure()
plt.scatter(y_prediction_for_plot_list, residuals_for_plot_list)
plt.title('Residuals vs Fitted Values')
plt.show()
'''

plt.figure()
plt.plot([1, 2, 3, 4, 5, 6], [3.077258216882004, 6.055762235851197, 9081.913555528949, 99.96292883377669, 74.79332635917763, 44.86165801948756], 'b-', linewidth = 2.0)
plt.title('RMSE vs Polynomial Degree for a fixed training and test set')
plt.show()

'''
plt.figure()
plt.plot([1, 2, 3, 4, 5, 6], [4.849841132519837, 3.4647278252138007, 3183.5537535780336, 147.61857479080868, 329.3302944474846, 141.55260699489529], 'b-', linewidth = 2.0)
plt.title('Average RMSE vs Polynomial Degree Using Cross Validation')
plt.show()
'''
