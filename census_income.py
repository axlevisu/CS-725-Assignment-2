#census_income
import numpy as np
import pandas
import csv
import math
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
# import ANN

# Parameters
nb_epoch = 10
al = 0.01
batch_size = 500
n_hidden = 100
train_data =[]
test_data =[]

# Read test and train data

file_name = "train.csv"
file = open(file_name, 'r')
input_attr = file.readline().strip().split(',')

for line in file:
	train_data.append(map(lambda x: x.strip(),list( line.strip().split(','))))

file_name = 'kaggle_test_data.csv'
file = open(file_name, 'r')
input_attr = file.readline().strip().split(',')

for line in file:
	test_data.append(map(lambda x: x.strip(),list( line.strip().split(','))))
# with open('train.csv') as f:
# 	file = csv.reader(f,delimiter=',')
# 		# ,quotechar ='|')
# 	for line in file:
# 		train_data.append(line)

# with open('kaggle_test_data.csv') as f:
# 	file = csv.reader(f,delimiter=',')
# 		# ,quotechar ='|')
# 	for line in file:
# 		test_data.append(line)
# del train_data[0]
# del test_data[0]
# train_data = np.array(train_data)[1:,1:]
# test_data = np.array(train_data)[1:,1:]

# Data Dictionary 

data_dict ={1:{'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3,
 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7},
3:{'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5, 
'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, 
'10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15},
5:{'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2,
'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6},
6:{'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4,
'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8,
'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13},
7:{'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5},
8:{'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4},
9:{'Female':0, 'Male':1},
13:{'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4,
 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 
 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 
 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24,
 'Laos':25, 'Ecuador':26, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32,
 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38,
 'Hong':39, 'Holand-Netherlands':40}
}
# Non-numeric indices
non_numeric =[1,3,5,6,7,8,9,13]
data_length = 15
# Preprocessing
X =[]
X_test =[]
for row in train_data:
	del row[0]
	data =[]
	for i in xrange(data_length):
		if i not in non_numeric:
			data.append(float(row[i]))
		else:
			value =[0]*len(data_dict[i])
			# if row[i] in data_dict[i]:
			if row[i] != '?':
				value[data_dict[i][row[i]]] = 1
			data =  data + value
	X.append(data) 


X = np.array(X)
Y = X[:,-1]
X = X[:,:-1]
# Y = np_utils.to_categorical(Y)
Nf = X.shape[1]
train_min =[]
train_std =[]
for i in range(Nf):
	f_min = np.min(X[:,i])
	f_std = np.std(X[:,i])
	train_min.append(f_min)
	train_std.append(f_std)
	temp = (X[:,i] - f_min)/f_std
	X[:,i] = temp


test_ids =[]
# Same for test data
data_length = 14
for row in test_data:
	test_ids.append(int(row[0]))
	del row[0]
	data =[]
	for i in xrange(data_length):
		if i not in non_numeric:
			data.append(float(row[i]))
		else:
			value =[0]*len(data_dict[i])
			# if row[i] in data_dict[i]:
			if row[i] != '?':
				value[data_dict[i][row[i]]] = 1
			data =  data + value
	X_test.append(data) 


X_test = np.array(X_test)
for i in range(Nf):
	temp = (X_test[:,i] - train_min[i])/train_std[i]
	X_test[:,i] = temp

# End of preprocessing

model = Sequential()
model.add(Dense(50, input_dim=Nf, kernel_initializer='normal', activation='relu'))
# model.add(Dense(500, kernel_initializer='normal', activation='relu'))
# model.add(Dense(32, init='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

Y_test = model.predict(X_test, batch_size=batch_size)
print Y_test[25:35]
output = [['id','salary']]
output = output + [[str(test_ids[i]),str(int(j>0.5))] for i,j in enumerate(Y_test.flatten())]

with open('output_keras.csv', 'w') as file:
	for line in output:
		file.write(",".join(line) + "\n")


error = model.evaluate(X, Y, batch_size=batch_size, verbose=1)
print "\n", error
