import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
import csv
import numpy as np
import statsmodels.api as sm
import math
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer

with open('O:\\UCLA 16W\\EE239\\Project1\\network_backup_dataset.csv') as f :
    r = list(csv.reader(f))

my_data = r[1 : ]
random.shuffle(my_data)

my_list_week = list()
my_list_day = list()
my_list_time = list()
my_list_work_flow = list()
my_list_file = list()
my_list_size = list()
my_list_backup_time = list()


# load data
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
    
X = (my_list_week,my_list_day,my_list_time,my_list_work_flow, my_list_file, my_list_backup_time,my_list_backup_time,my_list_size)
X=np.mat(X).T


x_test = X[:1859,0:-2]
y_test = X[:1859,-1]

x_train = X[1859:,0:-2]
y_train = X[1859:,-1]

my_list_prediction = list()
#input_size = x_train.shape[1]  #column number
#target_size = y_train.shape[1]
input_size = 6
target_size = 1
hidden0_size = 10
hidden1_size = 3
epochs = 1000
mse = 0

#prepare dataset
ds = SDS(input_size,target_size)
ds.setField('input',x_train)
ds.setField('target',y_train)

#init and train

net = buildNetwork(input_size, hidden1_size, target_size, bias = True,hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
#trainer.trainUntilConvergence(maxEpochs = 1000)

for i in range( epochs ):
	mse_t = trainer.train()
	rmse_t = sqrt( mse_t )
	print ('training RMSE, epoch {}: {}'.format( i + 1, rmse_t ))

for i in range(1859):
        y_prediction = net.activate(np.array(x_test[i,:])[0])
        my_list_prediction.append(y_prediction)
        mse = mse + (y_prediction - y_test[i,0])*(y_prediction - y_test[i,0])
    
mse=mse/1858
rmse=np.sqrt(mse)
print(rmse)
x = range(0, 1859)
#plt.plot(my_list_prediction,label="$Prediction Value$",color="red")
plt.scatter(x,my_list_prediction,color='r', marker = 'x',label="$Prediction Value$")
plt.hold(True)
plt.grid(True)
#plt.plot(my_list_size[:1859],"g",label="$Actual Value$")
plt.scatter(x,my_list_size[:1859],color = 'g',marker = '+',label="$Actual Value$")
plt.title('Prediction by Neural Network Regression Model')
plt.ylabel('Backup Size/GB')
plt.xlabel('File Number')
plt.axis([0,2000,-0.2,1.2])
plt.legend()
plt.show()
