######## Duplication Typecast ####
import pandas as pd

data = pd.read_csv("F:\data sci Assignment\data prepro csv file\online retail1.csv",encoding='unicode_escape')
data
data.info()
## only unit price has float value####
data.UnitPrice  = data.UnitPrice.astype('int64') 
data.dtypes
data

##### find duplicate value
duplicate = data.duplicated()
duplicate
sum(duplicate)

data1 = data.drop_duplicates() 
data1

data2 = data1.duplicated()
sum(data2)
data2


####3  eda 
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import numpy as np
import seaborn as sns


data.boxplot('UnitPrice')   ##### in unit price have outliers 

IQR = data['UnitPrice'].quantile(0.75) - data['UnitPrice'].quantile(0.25)
IQR
lower_limit = data['UnitPrice'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['UnitPrice'].quantile(0.75) + (IQR * 1.5)
upper_limit

data['new_Unit_price'] = pd.DataFrame(np.where(data['UnitPrice'] > upper_limit, upper_limit, np.where(data['UnitPrice'] < lower_limit, lower_limit, data['UnitPrice'])))

sns.boxplot(data.new_Unit_price);plt.title('Boxplot');plt.show()



data.hist('UnitPrice')
stats.probplot(data.UnitPrice, dist="norm", plot=pylab)
plt.scatter(data['UnitPrice'],data['UnitPrice'])
