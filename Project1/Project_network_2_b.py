import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import cross_validation

with open('C:\\Users\\Gaoxiang\\Desktop\\EE239\\network_backup_dataset.csv', newline = '') as f :
     r = list(csv.reader(f))

my_data = r[1 : ]
##random.shuffle(my_data)

my_list_week = list()
my_list_day = list()
my_list_time = list()
my_list_work_flow = list()
my_list_file = list()
my_list_size = list()
my_list_backup_time = list()



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

y = my_list_size[:]
x1=my_list_week[:]
x2=my_list_day[:]
x3=my_list_time[:]
x4=my_list_work_flow[:]
x5=my_list_file[:]
x6=my_list_backup_time[:]

X = [x1, x2, x3, x4, x5, x6]
X=np.asarray(X).T
list_rmse=list()

##for depth in range(4,11):
##for estimator in [1,10,15,20]:
##    regressor = RandomForestRegressor(n_estimators=estimator, max_features=6, max_depth=10)
##    scores=cross_validation.cross_val_score(regressor,X,y,cv=10, scoring='mean_squared_error')
##    y_predict=cross_validation.cross_val_predict(regressor,X,y,cv=10)
##    rmse=math.sqrt(-np.mean(scores))
##    list_rmse.append(rmse)
##    print(rmse)

##plt.scatter([1,10,15,20],list_rmse,color='r')
##plt.hold(True)
##plt.xlabel('Number of trees')
##plt.ylabel('RMSE')
##plt.title('Fitted Values vs Actual Values')
##plt.axis([0, 20, 0, 0.10])
##plt.show()

##plt.scatter(range(4,11),list_rmse,color='r')
##plt.hold(True)
##plt.show()


regressor = RandomForestRegressor(n_estimators=20, max_features=6, max_depth=10)
scores=cross_validation.cross_val_score(regressor,X,y,cv=10, scoring='mean_squared_error')
y_predict=cross_validation.cross_val_predict(regressor,X,y,cv=10)
rmse=math.sqrt(-np.mean(scores))
list_rmse.append(rmse)

plt.scatter(my_list_size,y_predict,marker='+',color='r')
plt.hold(True)
plt.plot([-0.2,1.2],[-0.2,1.2])
plt.axis([-0.2,1.2, -0.2, 1.2])
plt.xlabel('Actual values')
plt.ylabel('Fitted values')
plt.title('Fitted Values vs Actual Values')
plt.show()


