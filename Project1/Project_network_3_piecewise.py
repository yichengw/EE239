import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import statsmodels.api as sm
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

with open('D:\\study\\Winter 2016\\Big Data\\Project 1\\network_backup_dataset.csv', newline = '') as f :
    r = list(csv.reader(f))

my_data = r[1 : ]
random.shuffle(my_data)

my_list_week_0 = list()
my_list_day_0 = list()
my_list_time_0 = list()
my_list_file_0 = list()
my_list_size_0 = list()
my_list_backup_time_0 = list()

my_list_week_1 = list()
my_list_day_1 = list()
my_list_time_1 = list()
my_list_file_1 = list()
my_list_size_1 = list()
my_list_backup_time_1 = list()

my_list_week_2 = list()
my_list_day_2 = list()
my_list_time_2 = list()
my_list_file_2 = list()
my_list_size_2 = list()
my_list_backup_time_2 = list()

my_list_week_3 = list()
my_list_day_3 = list()
my_list_time_3 = list()
my_list_file_3 = list()
my_list_size_3 = list()
my_list_backup_time_3 = list()

my_list_week_4 = list()
my_list_day_4 = list()
my_list_time_4 = list()
my_list_file_4 = list()
my_list_size_4 = list()
my_list_backup_time_4 = list()

i = 0
while i <= len(my_data) - 1 :
        
    if my_data[i][3] == 'work_flow_0' :
        my_list_week_0.append(float(my_data[i][0]))
        my_list_size_0.append(float(my_data[i][5]))
        my_list_backup_time_0.append(float(my_data[i][6]))
        my_list_time_0.append(float(my_data[i][2]))
        
        if my_data[i][1] == 'Monday' :
            my_list_day_0.append(1)
        elif my_data[i][1] == 'Tuesday' :
            my_list_day_0.append(2)
        elif my_data[i][1] == 'Wednesday' :
            my_list_day_0.append(3)
        elif my_data[i][1] == 'Thursday' :
            my_list_day_0.append(4)
        elif my_data[i][1] == 'Friday' :
            my_list_day_0.append(5)
        elif my_data[i][1] == 'Saturday' :
            my_list_day_0.append(6)
        elif my_data[i][1] == 'Sunday' :
            my_list_day_0.append(7)

        if my_data[i][4] == 'File_0' :
            my_list_file_0.append(1)
        elif my_data[i][4] == 'File_1' :
            my_list_file_0.append(2)
        elif my_data[i][4] == 'File_2' :
            my_list_file_0.append(3)
        elif my_data[i][4] == 'File_3' :
            my_list_file_0.append(4)
        elif my_data[i][4] == 'File_4' :
            my_list_file_0.append(5)
        elif my_data[i][4] == 'File_5' :
            my_list_file_0.append(6)
        elif my_data[i][4] == 'File_6' :
            my_list_file_0.append(7)
        elif my_data[i][4] == 'File_7' :
            my_list_file_0.append(8)
        elif my_data[i][4] == 'File_8' :
            my_list_file_0.append(9)
        elif my_data[i][4] == 'File_9' :
            my_list_file_0.append(10)
        elif my_data[i][4] == 'File_10' :
            my_list_file_0.append(11)
        elif my_data[i][4] == 'File_11' :
            my_list_file_0.append(12)
        elif my_data[i][4] == 'File_12' :
            my_list_file_0.append(13)
        elif my_data[i][4] == 'File_13' :
            my_list_file_0.append(14)
        elif my_data[i][4] == 'File_14' :
            my_list_file_0.append(15)
        elif my_data[i][4] == 'File_15' :
            my_list_file_0.append(16)
        elif my_data[i][4] == 'File_16' :
            my_list_file_0.append(17)
        elif my_data[i][4] == 'File_17' :
            my_list_file_0.append(18)
        elif my_data[i][4] == 'File_18' :
            my_list_file_0.append(19)
        elif my_data[i][4] == 'File_19' :
            my_list_file_0.append(20)
        elif my_data[i][4] == 'File_20' :
            my_list_file_0.append(21)
        elif my_data[i][4] == 'File_21' :
            my_list_file_0.append(22)
        elif my_data[i][4] == 'File_22' :
            my_list_file_0.append(23)
        elif my_data[i][4] == 'File_23' :
            my_list_file_0.append(24)
        elif my_data[i][4] == 'File_24' :
            my_list_file_0.append(25)
        elif my_data[i][4] == 'File_25' :
            my_list_file_0.append(26)
        elif my_data[i][4] == 'File_26' :
            my_list_file_0.append(27)
        elif my_data[i][4] == 'File_27' :
            my_list_file_0.append(28)
        elif my_data[i][4] == 'File_28' :
            my_list_file_0.append(29)
        elif my_data[i][4] == 'File_29' :
            my_list_file_0.append(30)
            
    elif my_data[i][3] == 'work_flow_1' :
        my_list_week_1.append(float(my_data[i][0]))
        my_list_size_1.append(float(my_data[i][5]))
        my_list_backup_time_1.append(float(my_data[i][6]))
        my_list_time_1.append(float(my_data[i][2]))

        if my_data[i][1] == 'Monday' :
            my_list_day_1.append(1)
        elif my_data[i][1] == 'Tuesday' :
            my_list_day_1.append(2)
        elif my_data[i][1] == 'Wednesday' :
            my_list_day_1.append(3)
        elif my_data[i][1] == 'Thursday' :
            my_list_day_1.append(4)
        elif my_data[i][1] == 'Friday' :
            my_list_day_1.append(5)
        elif my_data[i][1] == 'Saturday' :
            my_list_day_1.append(6)
        elif my_data[i][1] == 'Sunday' :
            my_list_day_1.append(7)

        if my_data[i][4] == 'File_0' :
            my_list_file_1.append(1)
        elif my_data[i][4] == 'File_1' :
            my_list_file_1.append(2)
        elif my_data[i][4] == 'File_2' :
            my_list_file_1.append(3)
        elif my_data[i][4] == 'File_3' :
            my_list_file_1.append(4)
        elif my_data[i][4] == 'File_4' :
            my_list_file_1.append(5)
        elif my_data[i][4] == 'File_5' :
            my_list_file_1.append(6)
        elif my_data[i][4] == 'File_6' :
            my_list_file_1.append(7)
        elif my_data[i][4] == 'File_7' :
            my_list_file_1.append(8)
        elif my_data[i][4] == 'File_8' :
            my_list_file_1.append(9)
        elif my_data[i][4] == 'File_9' :
            my_list_file_1.append(10)
        elif my_data[i][4] == 'File_10' :
            my_list_file_1.append(11)
        elif my_data[i][4] == 'File_11' :
            my_list_file_1.append(12)
        elif my_data[i][4] == 'File_12' :
            my_list_file_1.append(13)
        elif my_data[i][4] == 'File_13' :
            my_list_file_1.append(14)
        elif my_data[i][4] == 'File_14' :
            my_list_file_1.append(15)
        elif my_data[i][4] == 'File_15' :
            my_list_file_1.append(16)
        elif my_data[i][4] == 'File_16' :
            my_list_file_1.append(17)
        elif my_data[i][4] == 'File_17' :
            my_list_file_1.append(18)
        elif my_data[i][4] == 'File_18' :
            my_list_file_1.append(19)
        elif my_data[i][4] == 'File_19' :
            my_list_file_1.append(20)
        elif my_data[i][4] == 'File_20' :
            my_list_file_1.append(21)
        elif my_data[i][4] == 'File_21' :
            my_list_file_1.append(22)
        elif my_data[i][4] == 'File_22' :
            my_list_file_1.append(23)
        elif my_data[i][4] == 'File_23' :
            my_list_file_1.append(24)
        elif my_data[i][4] == 'File_24' :
            my_list_file_1.append(25)
        elif my_data[i][4] == 'File_25' :
            my_list_file_1.append(26)
        elif my_data[i][4] == 'File_26' :
            my_list_file_1.append(27)
        elif my_data[i][4] == 'File_27' :
            my_list_file_1.append(28)
        elif my_data[i][4] == 'File_28' :
            my_list_file_1.append(29)
        elif my_data[i][4] == 'File_29' :
            my_list_file_1.append(30)
        
    elif my_data[i][3] == 'work_flow_2' :
        my_list_week_2.append(float(my_data[i][0]))
        my_list_size_2.append(float(my_data[i][5]))
        my_list_backup_time_2.append(float(my_data[i][6]))
        my_list_time_2.append(float(my_data[i][2]))

        if my_data[i][1] == 'Monday' :
            my_list_day_2.append(1)
        elif my_data[i][1] == 'Tuesday' :
            my_list_day_2.append(2)
        elif my_data[i][1] == 'Wednesday' :
            my_list_day_2.append(3)
        elif my_data[i][1] == 'Thursday' :
            my_list_day_2.append(4)
        elif my_data[i][1] == 'Friday' :
            my_list_day_2.append(5)
        elif my_data[i][1] == 'Saturday' :
            my_list_day_2.append(6)
        elif my_data[i][1] == 'Sunday' :
            my_list_day_2.append(7)

        if my_data[i][4] == 'File_0' :
            my_list_file_2.append(1)
        elif my_data[i][4] == 'File_1' :
            my_list_file_2.append(2)
        elif my_data[i][4] == 'File_2' :
            my_list_file_2.append(3)
        elif my_data[i][4] == 'File_3' :
            my_list_file_2.append(4)
        elif my_data[i][4] == 'File_4' :
            my_list_file_2.append(5)
        elif my_data[i][4] == 'File_5' :
            my_list_file_2.append(6)
        elif my_data[i][4] == 'File_6' :
            my_list_file_2.append(7)
        elif my_data[i][4] == 'File_7' :
            my_list_file_2.append(8)
        elif my_data[i][4] == 'File_8' :
            my_list_file_2.append(9)
        elif my_data[i][4] == 'File_9' :
            my_list_file_2.append(10)
        elif my_data[i][4] == 'File_10' :
            my_list_file_2.append(11)
        elif my_data[i][4] == 'File_11' :
            my_list_file_2.append(12)
        elif my_data[i][4] == 'File_12' :
            my_list_file_2.append(13)
        elif my_data[i][4] == 'File_13' :
            my_list_file_2.append(14)
        elif my_data[i][4] == 'File_14' :
            my_list_file_2.append(15)
        elif my_data[i][4] == 'File_15' :
            my_list_file_2.append(16)
        elif my_data[i][4] == 'File_16' :
            my_list_file_2.append(17)
        elif my_data[i][4] == 'File_17' :
            my_list_file_2.append(18)
        elif my_data[i][4] == 'File_18' :
            my_list_file_2.append(19)
        elif my_data[i][4] == 'File_19' :
            my_list_file_2.append(20)
        elif my_data[i][4] == 'File_20' :
            my_list_file_2.append(21)
        elif my_data[i][4] == 'File_21' :
            my_list_file_2.append(22)
        elif my_data[i][4] == 'File_22' :
            my_list_file_2.append(23)
        elif my_data[i][4] == 'File_23' :
            my_list_file_2.append(24)
        elif my_data[i][4] == 'File_24' :
            my_list_file_2.append(25)
        elif my_data[i][4] == 'File_25' :
            my_list_file_2.append(26)
        elif my_data[i][4] == 'File_26' :
            my_list_file_2.append(27)
        elif my_data[i][4] == 'File_27' :
            my_list_file_2.append(28)
        elif my_data[i][4] == 'File_28' :
            my_list_file_2.append(29)
        elif my_data[i][4] == 'File_29' :
            my_list_file_2.append(30)
        
    elif my_data[i][3] == 'work_flow_3' :
        my_list_week_3.append(float(my_data[i][0]))
        my_list_size_3.append(float(my_data[i][5]))
        my_list_backup_time_3.append(float(my_data[i][6]))
        my_list_time_3.append(float(my_data[i][2]))

        if my_data[i][1] == 'Monday' :
            my_list_day_3.append(1)
        elif my_data[i][1] == 'Tuesday' :
            my_list_day_3.append(2)
        elif my_data[i][1] == 'Wednesday' :
            my_list_day_3.append(3)
        elif my_data[i][1] == 'Thursday' :
            my_list_day_3.append(4)
        elif my_data[i][1] == 'Friday' :
            my_list_day_3.append(5)
        elif my_data[i][1] == 'Saturday' :
            my_list_day_3.append(6)
        elif my_data[i][1] == 'Sunday' :
            my_list_day_3.append(7)

        if my_data[i][4] == 'File_0' :
            my_list_file_3.append(1)
        elif my_data[i][4] == 'File_1' :
            my_list_file_3.append(2)
        elif my_data[i][4] == 'File_2' :
            my_list_file_3.append(3)
        elif my_data[i][4] == 'File_3' :
            my_list_file_3.append(4)
        elif my_data[i][4] == 'File_4' :
            my_list_file_3.append(5)
        elif my_data[i][4] == 'File_5' :
            my_list_file_3.append(6)
        elif my_data[i][4] == 'File_6' :
            my_list_file_3.append(7)
        elif my_data[i][4] == 'File_7' :
            my_list_file_3.append(8)
        elif my_data[i][4] == 'File_8' :
            my_list_file_3.append(9)
        elif my_data[i][4] == 'File_9' :
            my_list_file_3.append(10)
        elif my_data[i][4] == 'File_10' :
            my_list_file_3.append(11)
        elif my_data[i][4] == 'File_11' :
            my_list_file_3.append(12)
        elif my_data[i][4] == 'File_12' :
            my_list_file_3.append(13)
        elif my_data[i][4] == 'File_13' :
            my_list_file_3.append(14)
        elif my_data[i][4] == 'File_14' :
            my_list_file_3.append(15)
        elif my_data[i][4] == 'File_15' :
            my_list_file_3.append(16)
        elif my_data[i][4] == 'File_16' :
            my_list_file_3.append(17)
        elif my_data[i][4] == 'File_17' :
            my_list_file_3.append(18)
        elif my_data[i][4] == 'File_18' :
            my_list_file_3.append(19)
        elif my_data[i][4] == 'File_19' :
            my_list_file_3.append(20)
        elif my_data[i][4] == 'File_20' :
            my_list_file_3.append(21)
        elif my_data[i][4] == 'File_21' :
            my_list_file_3.append(22)
        elif my_data[i][4] == 'File_22' :
            my_list_file_3.append(23)
        elif my_data[i][4] == 'File_23' :
            my_list_file_3.append(24)
        elif my_data[i][4] == 'File_24' :
            my_list_file_3.append(25)
        elif my_data[i][4] == 'File_25' :
            my_list_file_3.append(26)
        elif my_data[i][4] == 'File_26' :
            my_list_file_3.append(27)
        elif my_data[i][4] == 'File_27' :
            my_list_file_3.append(28)
        elif my_data[i][4] == 'File_28' :
            my_list_file_3.append(29)
        elif my_data[i][4] == 'File_29' :
            my_list_file_3.append(30)
            
    elif my_data[i][3] == 'work_flow_4' :
        my_list_week_4.append(float(my_data[i][0]))
        my_list_size_4.append(float(my_data[i][5]))
        my_list_backup_time_4.append(float(my_data[i][6]))
        my_list_time_4.append(float(my_data[i][2]))

        if my_data[i][1] == 'Monday' :
            my_list_day_4.append(1)
        elif my_data[i][1] == 'Tuesday' :
            my_list_day_4.append(2)
        elif my_data[i][1] == 'Wednesday' :
            my_list_day_4.append(3)
        elif my_data[i][1] == 'Thursday' :
            my_list_day_4.append(4)
        elif my_data[i][1] == 'Friday' :
            my_list_day_4.append(5)
        elif my_data[i][1] == 'Saturday' :
            my_list_day_4.append(6)
        elif my_data[i][1] == 'Sunday' :
            my_list_day_4.append(7)

        if my_data[i][4] == 'File_0' :
            my_list_file_4.append(1)
        elif my_data[i][4] == 'File_1' :
            my_list_file_4.append(2)
        elif my_data[i][4] == 'File_2' :
            my_list_file_4.append(3)
        elif my_data[i][4] == 'File_3' :
            my_list_file_4.append(4)
        elif my_data[i][4] == 'File_4' :
            my_list_file_4.append(5)
        elif my_data[i][4] == 'File_5' :
            my_list_file_4.append(6)
        elif my_data[i][4] == 'File_6' :
            my_list_file_4.append(7)
        elif my_data[i][4] == 'File_7' :
            my_list_file_4.append(8)
        elif my_data[i][4] == 'File_8' :
            my_list_file_4.append(9)
        elif my_data[i][4] == 'File_9' :
            my_list_file_4.append(10)
        elif my_data[i][4] == 'File_10' :
            my_list_file_4.append(11)
        elif my_data[i][4] == 'File_11' :
            my_list_file_4.append(12)
        elif my_data[i][4] == 'File_12' :
            my_list_file_4.append(13)
        elif my_data[i][4] == 'File_13' :
            my_list_file_4.append(14)
        elif my_data[i][4] == 'File_14' :
            my_list_file_4.append(15)
        elif my_data[i][4] == 'File_15' :
            my_list_file_4.append(16)
        elif my_data[i][4] == 'File_16' :
            my_list_file_4.append(17)
        elif my_data[i][4] == 'File_17' :
            my_list_file_4.append(18)
        elif my_data[i][4] == 'File_18' :
            my_list_file_4.append(19)
        elif my_data[i][4] == 'File_19' :
            my_list_file_4.append(20)
        elif my_data[i][4] == 'File_20' :
            my_list_file_4.append(21)
        elif my_data[i][4] == 'File_21' :
            my_list_file_4.append(22)
        elif my_data[i][4] == 'File_22' :
            my_list_file_4.append(23)
        elif my_data[i][4] == 'File_23' :
            my_list_file_4.append(24)
        elif my_data[i][4] == 'File_24' :
            my_list_file_4.append(25)
        elif my_data[i][4] == 'File_25' :
            my_list_file_4.append(26)
        elif my_data[i][4] == 'File_26' :
            my_list_file_4.append(27)
        elif my_data[i][4] == 'File_27' :
            my_list_file_4.append(28)
        elif my_data[i][4] == 'File_28' :
            my_list_file_4.append(29)
        elif my_data[i][4] == 'File_29' :
            my_list_file_4.append(30)

    i = i + 1
    
L_0=len(my_list_week_0)
L_1=len(my_list_week_1)
L_2=len(my_list_week_2)
L_3=len(my_list_week_3)
L_4=len(my_list_week_4)
reg = LinearRegression()

for k in range(5) :
#    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8] :
        if k == 0 :
            y_cp = my_list_size_0[:]
            x_week_cp = my_list_week_0[:]
            x_day_cp = my_list_day_0[:]
            x_time_cp = my_list_time_0[:]
            x_file_cp = my_list_file_0[:]
            x_backup_time_cp = my_list_backup_time_0[:]
            L_k=L_0
        elif k == 1 :
            y_cp = my_list_size_1[:]
            x_week_cp = my_list_week_1[:]
            x_day_cp = my_list_day_1[:]
            x_time_cp = my_list_time_1[:]
            x_file_cp = my_list_file_1[:]
            x_backup_time_cp = my_list_backup_time_1[:]
            L_k=L_1
        elif k == 2 :
            y_cp = my_list_size_2[:]
            x_week_cp = my_list_week_2[:]
            x_day_cp = my_list_day_2[:]
            x_time_cp = my_list_time_2[:]
            x_file_cp = my_list_file_2[:]
            x_backup_time_cp = my_list_backup_time_2[:]
            L_k=L_2
        elif k == 3 :
            y_cp = my_list_size_3[:]
            x_week_cp = my_list_week_3[:]
            x_day_cp = my_list_day_3[:]
            x_time_cp = my_list_time_3[:]
            x_file_cp = my_list_file_3[:]
            x_backup_time_cp = my_list_backup_time_3[:]
            L_k=L_3
        elif k == 4 :
            y_cp = my_list_size_4[:]
            x_week_cp = my_list_week_4[:]
            x_day_cp = my_list_day_4[:]
            x_time_cp = my_list_time_4[:]
            x_file_cp = my_list_file_4[:]
            x_backup_time_cp = my_list_backup_time_4[:]
            L_k=L_4
            
        mse = 0
        y1 = y_cp
        x1 = [x_week_cp, x_day_cp, x_time_cp, x_file_cp, x_backup_time_cp]

        y1_a = np.asarray(y1)
        x1_a = np.asarray(x1).T

        poly = PolynomialFeatures(1)
        x1_p = poly.fit_transform(x1_a)
        y1_p = y1_a
    
        reg.fit(x1_a, y1_a)
        scores=cross_validation.cross_val_score(reg,x1_a,y1_a,scoring='mean_squared_error')
        print (math.sqrt((-np.mean(scores))))

