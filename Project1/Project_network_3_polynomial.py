import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

with open('D:\\study\\Winter 2016\\Big Data\\Project 1\\network_backup_dataset.csv', newline = '') as f :
     r = list(csv.reader(f))

my_data = r[1 : ]
my_data_1 = r[1 : ]
random.shuffle(my_data)

my_list_week = list()
my_list_day = list()
my_list_time = list()
my_list_work_flow = list()
my_list_file = list()
my_list_size = list()
my_list_backup_time = list()

my_list_week_1 = list()
my_list_day_1 = list()
my_list_time_1 = list()
my_list_work_flow_1 = list()
my_list_file_1 = list()
my_list_size_1 = list()
my_list_backup_time_1 = list()

i = 0
while i <= len(my_data) - 1 :
    my_list_size.append(float(my_data[i][5]))

    my_list_backup_time.append(float(my_data[i][6]))
    
    if my_data[i][0] == '1' :
        my_list_week.append(1)
    elif my_data[i][0] == '2' :
        my_list_week.append(2)
    elif my_data[i][0] == '3' :
        my_list_week.append(3)
    elif my_data[i][0] == '4' :
        my_list_week.append(4)
    elif my_data[i][0] == '5' :
        my_list_week.append(5)
    elif my_data[i][0] == '6' :
        my_list_week.append(6)
    elif my_data[i][0] == '7' :
        my_list_week.append(7)
    elif my_data[i][0] == '8' :
        my_list_week.append(8)
    elif my_data[i][0] == '9' :
        my_list_week.append(9)
    elif my_data[i][0] == '10' :
        my_list_week.append(10)
    elif my_data[i][0] == '11' :
        my_list_week.append(11)
    elif my_data[i][0] == '12' :
        my_list_week.append(12)
    elif my_data[i][0] == '13' :
        my_list_week.append(13)
    elif my_data[i][0] == '14' :
        my_list_week.append(14)
    elif my_data[i][0] == '15' :
        my_list_week.append(15)

    if my_data[i][1] == 'Monday' :
        my_list_day.append(1)
    elif my_data[i][1] == 'Tuesday' :
        my_list_day.append(2)
    elif my_data[i][1] == 'Wednesday' :
        my_list_day.append(3)
    elif my_data[i][1] == 'Thursday' :
        my_list_day.append(4)
    elif my_data[i][1] == 'Friday' :
        my_list_day.append(5)
    elif my_data[i][1] == 'Saturday' :
        my_list_day.append(6)
    elif my_data[i][1] == 'Sunday' :
        my_list_day.append(7)

    if my_data[i][2] == '1' :
        my_list_time.append(1)
    elif my_data[i][2] == '5' :
        my_list_time.append(5)
    elif my_data[i][2] == '9' :
        my_list_time.append(9)
    elif my_data[i][2] == '13' :
        my_list_time.append(13)
    elif my_data[i][2] == '17' :
        my_list_time.append(17)
    elif my_data[i][2] == '21' :
        my_list_time.append(21)
        
    if my_data[i][3] == 'work_flow_0' :
        my_list_work_flow.append(1)
    elif my_data[i][3] == 'work_flow_1' :
        my_list_work_flow.append(2)
    elif my_data[i][3] == 'work_flow_2' :
        my_list_work_flow.append(3)
    elif my_data[i][3] == 'work_flow_3' :
        my_list_work_flow.append(4)
    elif my_data[i][3] == 'work_flow_4' :
        my_list_work_flow.append(5)

    if my_data[i][4] == 'File_0' :
        my_list_file.append(1)
    elif my_data[i][4] == 'File_1' :
        my_list_file.append(2)
    elif my_data[i][4] == 'File_2' :
        my_list_file.append(3)
    elif my_data[i][4] == 'File_3' :
        my_list_file.append(4)
    elif my_data[i][4] == 'File_4' :
        my_list_file.append(5)
    elif my_data[i][4] == 'File_5' :
        my_list_file.append(6)
    elif my_data[i][4] == 'File_6' :
        my_list_file.append(7)
    elif my_data[i][4] == 'File_7' :
        my_list_file.append(8)
    elif my_data[i][4] == 'File_8' :
        my_list_file.append(9)
    elif my_data[i][4] == 'File_9' :
        my_list_file.append(10)
    elif my_data[i][4] == 'File_10' :
        my_list_file.append(11)
    elif my_data[i][4] == 'File_11' :
        my_list_file.append(12)
    elif my_data[i][4] == 'File_12' :
        my_list_file.append(13)
    elif my_data[i][4] == 'File_13' :
        my_list_file.append(14)
    elif my_data[i][4] == 'File_14' :
        my_list_file.append(15)
    elif my_data[i][4] == 'File_15' :
        my_list_file.append(16)
    elif my_data[i][4] == 'File_16' :
        my_list_file.append(17)
    elif my_data[i][4] == 'File_17' :
        my_list_file.append(18)
    elif my_data[i][4] == 'File_18' :
        my_list_file.append(19)
    elif my_data[i][4] == 'File_19' :
        my_list_file.append(20)
    elif my_data[i][4] == 'File_20' :
        my_list_file.append(21)
    elif my_data[i][4] == 'File_21' :
        my_list_file.append(22)
    elif my_data[i][4] == 'File_22' :
        my_list_file.append(23)
    elif my_data[i][4] == 'File_23' :
        my_list_file.append(24)
    elif my_data[i][4] == 'File_24' :
        my_list_file.append(25)
    elif my_data[i][4] == 'File_25' :
        my_list_file.append(26)
    elif my_data[i][4] == 'File_26' :
        my_list_file.append(27)
    elif my_data[i][4] == 'File_27' :
        my_list_file.append(28)
    elif my_data[i][4] == 'File_28' :
        my_list_file.append(29)
    elif my_data[i][4] == 'File_29' :
        my_list_file.append(30)

    i = i + 1

i = 0
while i <= len(my_data_1) - 1 :
    my_list_size_1.append(float(my_data_1[i][5]))

    my_list_backup_time_1.append(float(my_data_1[i][6]))
    
    if my_data_1[i][0] == '1' :
        my_list_week_1.append(1)
    elif my_data_1[i][0] == '2' :
        my_list_week_1.append(2)
    elif my_data_1[i][0] == '3' :
        my_list_week_1.append(3)
    elif my_data_1[i][0] == '4' :
        my_list_week_1.append(4)
    elif my_data_1[i][0] == '5' :
        my_list_week_1.append(5)
    elif my_data_1[i][0] == '6' :
        my_list_week_1.append(6)
    elif my_data_1[i][0] == '7' :
        my_list_week_1.append(7)
    elif my_data_1[i][0] == '8' :
        my_list_week_1.append(8)
    elif my_data_1[i][0] == '9' :
        my_list_week_1.append(9)
    elif my_data_1[i][0] == '10' :
        my_list_week_1.append(10)
    elif my_data_1[i][0] == '11' :
        my_list_week_1.append(11)
    elif my_data_1[i][0] == '12' :
        my_list_week_1.append(12)
    elif my_data_1[i][0] == '13' :
        my_list_week_1.append(13)
    elif my_data_1[i][0] == '14' :
        my_list_week_1.append(14)
    elif my_data_1[i][0] == '15' :
        my_list_week_1.append(15)

    if my_data_1[i][1] == 'Monday' :
        my_list_day_1.append(1)
    elif my_data_1[i][1] == 'Tuesday' :
        my_list_day_1.append(2)
    elif my_data_1[i][1] == 'Wednesday' :
        my_list_day_1.append(3)
    elif my_data_1[i][1] == 'Thursday' :
        my_list_day_1.append(4)
    elif my_data_1[i][1] == 'Friday' :
        my_list_day_1.append(5)
    elif my_data_1[i][1] == 'Saturday' :
        my_list_day_1.append(6)
    elif my_data_1[i][1] == 'Sunday' :
        my_list_day_1.append(7)

    if my_data_1[i][2] == '1' :
        my_list_time_1.append(1)
    elif my_data_1[i][2] == '5' :
        my_list_time_1.append(5)
    elif my_data_1[i][2] == '9' :
        my_list_time_1.append(9)
    elif my_data_1[i][2] == '13' :
        my_list_time_1.append(13)
    elif my_data_1[i][2] == '17' :
        my_list_time_1.append(17)
    elif my_data_1[i][2] == '21' :
        my_list_time_1.append(21)
        
    if my_data_1[i][3] == 'work_flow_0' :
        my_list_work_flow_1.append(1)
    elif my_data_1[i][3] == 'work_flow_1' :
        my_list_work_flow_1.append(2)
    elif my_data_1[i][3] == 'work_flow_2' :
        my_list_work_flow_1.append(3)
    elif my_data_1[i][3] == 'work_flow_3' :
        my_list_work_flow_1.append(4)
    elif my_data_1[i][3] == 'work_flow_4' :
        my_list_work_flow_1.append(5)

    if my_data_1[i][4] == 'File_0' :
        my_list_file_1.append(1)
    elif my_data_1[i][4] == 'File_1' :
        my_list_file_1.append(2)
    elif my_data_1[i][4] == 'File_2' :
        my_list_file_1.append(3)
    elif my_data_1[i][4] == 'File_3' :
        my_list_file_1.append(4)
    elif my_data_1[i][4] == 'File_4' :
        my_list_file_1.append(5)
    elif my_data_1[i][4] == 'File_5' :
        my_list_file_1.append(6)
    elif my_data_1[i][4] == 'File_6' :
        my_list_file_1.append(7)
    elif my_data_1[i][4] == 'File_7' :
        my_list_file_1.append(8)
    elif my_data_1[i][4] == 'File_8' :
        my_list_file_1.append(9)
    elif my_data_1[i][4] == 'File_9' :
        my_list_file_1.append(10)
    elif my_data_1[i][4] == 'File_10' :
        my_list_file_1.append(11)
    elif my_data_1[i][4] == 'File_11' :
        my_list_file_1.append(12)
    elif my_data_1[i][4] == 'File_12' :
        my_list_file_1.append(13)
    elif my_data_1[i][4] == 'File_13' :
        my_list_file_1.append(14)
    elif my_data_1[i][4] == 'File_14' :
        my_list_file_1.append(15)
    elif my_data_1[i][4] == 'File_15' :
        my_list_file_1.append(16)
    elif my_data_1[i][4] == 'File_16' :
        my_list_file_1.append(17)
    elif my_data_1[i][4] == 'File_17' :
        my_list_file_1.append(18)
    elif my_data_1[i][4] == 'File_18' :
        my_list_file_1.append(19)
    elif my_data_1[i][4] == 'File_19' :
        my_list_file_1.append(20)
    elif my_data_1[i][4] == 'File_20' :
        my_list_file_1.append(21)
    elif my_data_1[i][4] == 'File_21' :
        my_list_file_1.append(22)
    elif my_data_1[i][4] == 'File_22' :
        my_list_file_1.append(23)
    elif my_data_1[i][4] == 'File_23' :
        my_list_file_1.append(24)
    elif my_data_1[i][4] == 'File_24' :
        my_list_file_1.append(25)
    elif my_data_1[i][4] == 'File_25' :
        my_list_file_1.append(26)
    elif my_data_1[i][4] == 'File_26' :
        my_list_file_1.append(27)
    elif my_data_1[i][4] == 'File_27' :
        my_list_file_1.append(28)
    elif my_data_1[i][4] == 'File_28' :
        my_list_file_1.append(29)
    elif my_data_1[i][4] == 'File_29' :
        my_list_file_1.append(30)

    i = i + 1

reg = LinearRegression()
poly = PolynomialFeatures(6)
rmse_average = 0


for j in [0, 1, 2, 3, 4, 5, 6, 7, 8] :
    i = j*1858
    mse = 0

    y_cp = my_list_size[:]
    x_week_cp = my_list_week[:]
    x_day_cp = my_list_day[:]
    x_time_cp = my_list_time[:]
    x_work_flow_cp = my_list_work_flow[:]
    x_file_cp = my_list_file[:]
    x_backup_time_cp = my_list_backup_time[:]

    del y_cp[j*1858:(j+1)*1858]
    del x_week_cp[j*1858:(j+1)*1858]
    del x_day_cp[j*1858:(j+1)*1858]
    del x_time_cp[j*1858:(j+1)*1858]
    del x_work_flow_cp[j*1858:(j+1)*1858]
    del x_file_cp[j*1858:(j+1)*1858]
    del x_backup_time_cp[j*1858:(j+1)*1858]
    
    y1 = y_cp
    x1 = [x_week_cp, x_day_cp, x_time_cp, x_work_flow_cp, x_file_cp, x_backup_time_cp]

    y1_a = np.asarray(y1)
    x1_a = np.asarray(x1).T

    x1_p = poly.fit_transform(x1_a)
    y1_p = y1_a

    reg.fit(x1_p, y1_p)
    #print coeff
    #print('coeff column')
    #print(reg.coef_)
    
    while i < 1858*(j+1) :
        x_t = np.asarray([my_list_week[i], my_list_day[i], my_list_time[i], my_list_work_flow[i], my_list_file[i], my_list_backup_time[i]]).reshape(1, -1)
        x_t_p = poly.fit_transform(x_t)
        y_prediction = reg.predict(x_t_p)
        y_prediction_list = y_prediction.tolist()[0]
        mse = mse + (y_prediction_list - my_list_size[i])**2
        i = i + 1
        
    rmse_average = rmse_average + math.sqrt(mse/1859)
    print(math.sqrt(mse/1859))

# 10th training
mse = 0
i = 9*1858
y_cp = my_list_size[:]
x_week_cp = my_list_week[:]
x_day_cp = my_list_day[:]
x_time_cp = my_list_time[:]
x_work_flow_cp = my_list_work_flow[:]
x_file_cp = my_list_file[:]
x_backup_time_cp = my_list_backup_time[:]
del y_cp[9*1858:]
del x_week_cp[9*1858:]
del x_day_cp[9*1858:]
del x_time_cp[9*1858:]
del x_work_flow_cp[9*1858:]
del x_file_cp[9*1858:]
del x_backup_time_cp[9*1858:]
y1 = y_cp
x1 = [x_week_cp, x_day_cp, x_time_cp, x_work_flow_cp, x_file_cp, x_backup_time_cp]
y1_a = np.asarray(y1)
x1_a = np.asarray(x1).T
x1_p = poly.fit_transform(x1_a)
y1_p = y1_a
reg.fit(x1_p, y1_p)
#print(reg.coef_)
fitted_list = list()

while i < 18588 :
    x_t = np.asarray([my_list_week[i], my_list_day[i], my_list_time[i], my_list_work_flow[i], my_list_file[i], my_list_backup_time[i]]).reshape(1, -1)
    x_t_p = poly.fit_transform(x_t)
    y_prediction = reg.predict(x_t_p)
    y_prediction_list = y_prediction.tolist()[0]
    fitted_list.append(y_prediction_list)
    mse = mse + (y_prediction_list - my_list_size[i])**2
    i = i + 1

rmse_average = rmse_average + math.sqrt(mse/1866)
print(math.sqrt(mse/1866))

print('the average rmse:')
print(rmse_average/10)



mse_1 = 0;

y_cp_1 = my_list_size_1[:]
x_week_cp_1 = my_list_week_1[:]
x_day_cp_1 = my_list_day_1[:]
x_time_cp_1 = my_list_time_1[:]
x_work_flow_cp_1 = my_list_work_flow_1[:]
x_file_cp_1 = my_list_file_1[:]
x_backup_time_cp_1 = my_list_backup_time_1[:]

del y_cp_1[0:1858]
del x_week_cp_1[0:1858]
del x_day_cp_1[0:1858]
del x_time_cp_1[0:1858]
del x_work_flow_cp_1[0:1858]
del x_file_cp_1[0:1858]
del x_backup_time_cp_1[0:1858]

y1_1 = y_cp_1
x1_1 = [x_week_cp_1, x_day_cp_1, x_time_cp_1, x_work_flow_cp_1, x_file_cp_1, x_backup_time_cp_1]
y1_a_1 = np.asarray(y1_1)
x1_a_1 = np.asarray(x1_1).T
x1_p_1 = poly.fit_transform(x1_a_1)
y1_p_1 = y1_a_1

reg.fit(x1_p_1, y1_p_1)

i = 0
while i < 1858 :
    x_t_1 = np.asarray([my_list_week_1[i], my_list_day_1[i], my_list_time_1[i], my_list_work_flow_1[i], my_list_file_1[i], my_list_backup_time_1[i]]).reshape(1, -1)
    x_t_p_1 = poly.fit_transform(x_t_1)
    y_prediction_1 = reg.predict(x_t_p_1)
    y_prediction_list_1 = y_prediction_1.tolist()[0]
    mse_1 = mse_1 + (y_prediction_list_1 - my_list_size_1[i])**2
    i = i + 1

print('rmse for fixed training set is:')
print(math.sqrt(mse_1/1859))



plt.figure()
plt.scatter(range(1866), my_list_size[9*1858:], marker = 'x', color = 'y')
plt.hold(True)
plt.scatter(range(1866), fitted_list, marker = '+', color = 'b')
wplt.axis([0, 2000, -0.2, 1.2])
plt.title('Fitted values and actual values scattered plot over time')
plt.show()

plt.figure()
plt.scatter(fitted_list, my_list_size[9*1858:])
plt.hold(True)
plt.plot([-0.2,1.3],[-0.2,1.3],'r-',linewidth = 2.0)
plt.axis([-0.2, 1.3, -0.2, 1.3])
plt.xlabel('Fitted values')
plt.ylabel('Actual values')
plt.title('Fitted Values vs Actual Values')
plt.show()

plt.figure()
plt.scatter((np.array(my_list_size[9*1858:]) - np.array(fitted_list)).tolist(), fitted_list)
plt.xlabel('Residual values')
plt.ylabel('Fitted values')
plt.title('Residual vs Fitted values plot')
plt.show()


plt.figure()
plt.plot([1, 2, 3, 4, 5, 6], [0.07945157806699031, 0.0697824904390024, 0.04956132521429152, 0.02772603276154891, 0.020330512175813203, 0.042715238472507085], 'b-', linewidth = 2.0)
plt.title('Average RMSE vs Polynomial Degree of Cross Validation')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE')
plt.show()

