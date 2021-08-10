import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("F:\data sci Assignment\data prepro csv file\cal.csv")
data

data.agg(['skew', 'kurtosis']).transpose()

data.hist(grid=False,
       figsize=(8, 6),
       bins=20)


########np.log

data.hist('Weight gained (grams)')
data.show

weight_data = data['Weight gained (grams)'].transform([np.sqrt , np.exp , np.log , np.reciprocal])
print(weight_data)



tran_weight = data['Weight gained (grams)'].transform([np.log]) 
print(tran_weight)
tran_weight.hist(bins = 20 ,layout = (2,2),edgecolor ='k',figsize =(10,8))
plt.suptitle("weight log output")
plt.show()
######np.sqrt
trans_weight = data['Weight gained (grams)'].transform([np.sqrt]) 
print(trans_weight)

trans_weight.hist(bins = 20 ,layout = (2,2),edgecolor ='k',figsize =(10,8))
plt.suptitle("weight sqrt output")
plt.show()
####np.exp and np.reciprocal  prsence of infinte value 
tran_weight2 = data['Weight gained (grams)'].transform([np.exp]) 
print(tran_weight2)

tran_weight2.hist(bins = 20 ,layout = (2,2),edgecolor ='k',figsize =(10,8))
plt.suptitle("weight exp output")
plt.show()
##################calories###########

cal_data  =  data['Calories Consumed'].transform([np.sqrt , np.exp , np.log , np.reciprocal])
print(cal_data)
tran_data = data['Calories Consumed'].transform([np.sqrt]) 
print(tran_data)

tran_data.hist(bins = 20 ,layout = (2,2),edgecolor ='k',figsize =(10,8))
plt.suptitle("cal sqrt output")
plt.show()

tran_data1 = data['Calories Consumed'].transform([np.log]) 
print(tran_data)

tran_data1.hist(bins = 20 ,layout = (2,2),edgecolor ='k',figsize =(10,8))
plt.suptitle("cal log output")
plt.show()