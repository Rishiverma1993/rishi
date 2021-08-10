import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler#The two most popular techniques for scaling numerical data prior to modeling are normalization and standardization.

### Standardization  ########
seed  = pd.read_csv("F:\data sci Assignment\data prepro csv file\seed.csv")
seed.describe()
#to show in dataframe

detail = seed.describe()
# To scale data
data_scaler = StandardScaler()



df = data_scaler.fit_transform(seed)

dataset = pd.DataFrame(df)
std_data = dataset.describe()


########## Normalization

seed_2 = pd.read_csv("F:\data sci Assignment\data prepro csv file\seed.csv")
detail_2 = seed_2.describe()

seed_2 = pd.get_dummies(seed_2, drop_first = True) #no values is there

#Custom Function
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)
seed2_norm = norm_func(seed_2)
norm = seed2_norm.describe()


#to check the data is norminal 

import scipy.stats as stats
import pylab


seed_2 = pd.read_csv("F:\data sci Assignment\data prepro csv file\seed.csv")
detail_2 = seed_2.describe()

stats.probplot(seed_2.Area, dist="norm", plot=pylab)
stats.probplot(seed_2.Area*seed_2.Area, dist="norm", plot=pylab) #good result of normalization
stats.probplot(np.sqrt(seed_2.Area), dist="norm", plot=pylab)
stats.probplot(np.log(seed_2.Area), dist="norm", plot=pylab)

