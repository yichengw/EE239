import csv
with open('/Users/Qiao/Documents/Graduate/UCLA/Courses/Winter 2016/EE 239AS/HW1/Data/network_backup_dataset.csv', newline = '') as f :
     r = list(csv.reader(f))
	 
my_list_0 = list('')
my_list_1 = list('')
my_list_2 = list('')
my_list_3 = list('')
my_list_4 = list('')
my_list_5 = list('')

my_list_6 = list('')
my_list_7 = list('')
my_list_8 = list('')
my_list_9 = list('')
my_list_10 = list('')
my_list_11 = list('')

my_list_12 = list('')
my_list_13 = list('')
my_list_14 = list('')
my_list_15 = list('')
my_list_16 = list('')
my_list_17 = list('')

my_list_18 = list('')
my_list_19 = list('')
my_list_20 = list('')
my_list_21 = list('')
my_list_22 = list('')
my_list_23 = list('')

my_list_24 = list('')
my_list_25 = list('')
my_list_26 = list('')
my_list_27 = list('')
my_list_28 = list('')
my_list_29 = list('')

x = 1
while x <= 18588 :
     if r[x][4] == 'File_0' :
          my_list_0.append(r[x][5])
     elif r[x][4] == 'File_1' :
          my_list_1.append(r[x][5])
     elif r[x][4] == 'File_2' :
          my_list_2.append(r[x][5])
     elif r[x][4] == 'File_3' :
          my_list_3.append(r[x][5])
     elif r[x][4] == 'File_4' :
          my_list_4.append(r[x][5])
     elif r[x][4] == 'File_5' :
          my_list_5.append(r[x][5])
     elif r[x][4] == 'File_6' :
          my_list_6.append(r[x][5])
     elif r[x][4] == 'File_7' :
          my_list_7.append(r[x][5])
     elif r[x][4] == 'File_8' :
          my_list_8.append(r[x][5])
     elif r[x][4] == 'File_9' :
          my_list_9.append(r[x][5])
     elif r[x][4] == 'File_10' :
          my_list_10.append(r[x][5])
     elif r[x][4] == 'File_11' :
          my_list_11.append(r[x][5])
     elif r[x][4] == 'File_12' :
          my_list_12.append(r[x][5])
     elif r[x][4] == 'File_13' :
          my_list_13.append(r[x][5])
     elif r[x][4] == 'File_14' :
          my_list_14.append(r[x][5])
     elif r[x][4] == 'File_15' :
          my_list_15.append(r[x][5])
     elif r[x][4] == 'File_16' :
          my_list_16.append(r[x][5])
     elif r[x][4] == 'File_17' :
          my_list_17.append(r[x][5])
     elif r[x][4] == 'File_18' :
          my_list_18.append(r[x][5])
     elif r[x][4] == 'File_19' :
          my_list_19.append(r[x][5])
     elif r[x][4] == 'File_20' :
          my_list_20.append(r[x][5])
     elif r[x][4] == 'File_21' :
          my_list_21.append(r[x][5])
     elif r[x][4] == 'File_22' :
          my_list_22.append(r[x][5])
     elif r[x][4] == 'File_23' :
          my_list_23.append(r[x][5])
     elif r[x][4] == 'File_24' :
          my_list_24.append(r[x][5])
     elif r[x][4] == 'File_25' :
          my_list_25.append(r[x][5])
     elif r[x][4] == 'File_26' :
          my_list_26.append(r[x][5])
     elif r[x][4] == 'File_27' :
          my_list_27.append(r[x][5])
     elif r[x][4] == 'File_28' :
          my_list_28.append(r[x][5])
     else :
          my_list_29.append(r[x][5])
     x = x + 1

my_list_num_0 = list()
my_list_num_1 = list()
my_list_num_2 = list()
my_list_num_3 = list()
my_list_num_4 = list()
my_list_num_5 = list()
my_list_num_6 = list()
my_list_num_7 = list()
my_list_num_8 = list()
my_list_num_9 = list()
my_list_num_10 = list()
my_list_num_11 = list()
my_list_num_12 = list()
my_list_num_13 = list()
my_list_num_14 = list()
my_list_num_15 = list()
my_list_num_16 = list()
my_list_num_17 = list()
my_list_num_18 = list()
my_list_num_19 = list()
my_list_num_20 = list()
my_list_num_21 = list()
my_list_num_22 = list()
my_list_num_23 = list()
my_list_num_24 = list()
my_list_num_25 = list()
my_list_num_26 = list()
my_list_num_27 = list()
my_list_num_28 = list()
my_list_num_29 = list()


x = 0
while x < len(my_list_0) :
     my_list_num_0.append(float(my_list_0[x]))
     x = x + 1

x = 0
while x < len(my_list_1) :
     my_list_num_1.append(float(my_list_1[x]))
     x = x + 1

x = 0
while x < len(my_list_2) :
     my_list_num_2.append(float(my_list_2[x]))
     x = x + 1

x = 0
while x < len(my_list_3) :
     my_list_num_3.append(float(my_list_3[x]))
     x = x + 1

x = 0
while x < len(my_list_4) :
     my_list_num_4.append(float(my_list_4[x]))
     x = x + 1

x = 0
while x < len(my_list_5) :
     my_list_num_5.append(float(my_list_5[x]))
     x = x + 1

x = 0
while x < len(my_list_6) :
     my_list_num_6.append(float(my_list_6[x]))
     x = x + 1


x = 0
while x < len(my_list_7) :
     my_list_num_7.append(float(my_list_7[x]))
     x = x + 1


x = 0
while x < len(my_list_8) :
     my_list_num_8.append(float(my_list_8[x]))
     x = x + 1


x = 0
while x < len(my_list_9) :
     my_list_num_9.append(float(my_list_9[x]))
     x = x + 1


x = 0
while x < len(my_list_10) :
     my_list_num_10.append(float(my_list_10[x]))
     x = x + 1


x = 0
while x < len(my_list_11) :
     my_list_num_11.append(float(my_list_11[x]))
     x = x + 1


x = 0
while x < len(my_list_12) :
     my_list_num_12.append(float(my_list_12[x]))
     x = x + 1

x = 0
while x < len(my_list_13) :
     my_list_num_13.append(float(my_list_13[x]))
     x = x + 1


x = 0
while x < len(my_list_14) :
     my_list_num_14.append(float(my_list_14[x]))
     x = x + 1


x = 0
while x < len(my_list_15) :
     my_list_num_15.append(float(my_list_15[x]))
     x = x + 1


x = 0
while x < len(my_list_16) :
     my_list_num_16.append(float(my_list_16[x]))
     x = x + 1

x = 0
while x < len(my_list_17) :
     my_list_num_17.append(float(my_list_17[x]))
     x = x + 1

x = 0
while x < len(my_list_18) :
     my_list_num_18.append(float(my_list_18[x]))
     x = x + 1


x = 0
while x < len(my_list_19) :
     my_list_num_19.append(float(my_list_19[x]))
     x = x + 1


x = 0
while x < len(my_list_20) :
     my_list_num_20.append(float(my_list_20[x]))
     x = x + 1


x = 0
while x < len(my_list_21) :
     my_list_num_21.append(float(my_list_21[x]))
     x = x + 1

x = 0
while x < len(my_list_22) :
     my_list_num_22.append(float(my_list_22[x]))
     x = x + 1

x = 0
while x < len(my_list_23) :
     my_list_num_23.append(float(my_list_23[x]))
     x = x + 1

x = 0
while x < len(my_list_24) :
     my_list_num_24.append(float(my_list_24[x]))
     x = x + 1


x = 0
while x < len(my_list_25) :
     my_list_num_25.append(float(my_list_25[x]))
     x = x + 1


x = 0
while x < len(my_list_26) :
     my_list_num_26.append(float(my_list_26[x]))
     x = x + 1

x = 0
while x < len(my_list_27) :
     my_list_num_27.append(float(my_list_27[x]))
     x = x + 1

x = 0
while x < len(my_list_28) :
     my_list_num_28.append(float(my_list_28[x]))
     x = x + 1

x = 0
while x < len(my_list_29) :
     my_list_num_29.append(float(my_list_29[x]))
     x = x + 1

#print(my_list_num)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(my_list_num_0, 'r-', linewidth = 2.0)
plt.hold(True)
plt.plot(my_list_num_1, 'g-', linewidth = 2.0)
plt.plot(my_list_num_2, 'b-', linewidth = 2.0)
plt.plot(my_list_num_3, 'c-', linewidth = 2.0)
plt.plot(my_list_num_4, 'm-', linewidth = 2.0)
plt.plot(my_list_num_5, 'y-', linewidth = 2.0)
plt.axis([0, 150, -0.1, 1.1])
plt.grid(True)
plt.title('Size of Backup of File 0-5 for Workflow 0')


plt.figure()
plt.plot(my_list_num_6, 'r-', linewidth = 2.0)
plt.hold(True)
plt.plot(my_list_num_7, 'g-', linewidth = 2.0)
plt.plot(my_list_num_8, 'b-', linewidth = 2.0)
plt.plot(my_list_num_9, 'c-', linewidth = 2.0)
plt.plot(my_list_num_10, 'm-', linewidth = 2.0)
plt.plot(my_list_num_11, 'y-', linewidth = 2.0)
plt.axis([0, 150, -0.1, 1.1])
plt.grid(True)
plt.title('Size of Backup of File 6-11 for Workflow 1')


plt.figure()
plt.plot(my_list_num_12, 'r-', linewidth = 2.0)
plt.hold(True)
plt.plot(my_list_num_13, 'g-', linewidth = 2.0)
plt.plot(my_list_num_14, 'b-', linewidth = 2.0)
plt.plot(my_list_num_15, 'c-', linewidth = 2.0)
plt.plot(my_list_num_16, 'm-', linewidth = 2.0)
plt.plot(my_list_num_17, 'y-', linewidth = 2.0)
plt.axis([0, 150, -0.1, 1.1])
plt.grid(True)
plt.title('Size of Backup of File 12-17 for Workflow 2')


plt.figure()
plt.plot(my_list_num_18, 'r-', linewidth = 2.0)
plt.hold(True)
plt.plot(my_list_num_19, 'g-', linewidth = 2.0)
plt.plot(my_list_num_20, 'b-', linewidth = 2.0)
plt.plot(my_list_num_21, 'c-', linewidth = 2.0)
plt.plot(my_list_num_22, 'm-', linewidth = 2.0)
plt.plot(my_list_num_23, 'y-', linewidth = 2.0)
plt.axis([0, 150, -0.1, 1.1])
plt.grid(True)
plt.title('Size of Backup of File 18-23 for Workflow 3')


plt.figure()
plt.plot(my_list_num_24, 'r-', linewidth = 2.0)
plt.hold(True)
plt.plot(my_list_num_25, 'g-', linewidth = 2.0)
plt.plot(my_list_num_26, 'b-', linewidth = 2.0)
plt.plot(my_list_num_27, 'c-', linewidth = 2.0)
plt.plot(my_list_num_28, 'm-', linewidth = 2.0)
plt.plot(my_list_num_29, 'y-', linewidth = 2.0)
plt.axis([0, 150, -0.1, 1.1])
plt.grid(True)
plt.title('Size of Backup of File 24-29 for Workflow 4')

plt.show()
