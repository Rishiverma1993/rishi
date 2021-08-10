import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv(r"F:\data sci Assignment\data prepro csv file\boostk.csv")
data.dtypes
################################
sns.boxplot(data.crim)

  # number of outlier is high 

IQR = data['crim'].quantile(0.75) - data['crim'].quantile(0.25)
IQR
lower_limit = data['crim'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['crim'].quantile(0.75) + (IQR * 1.5)
upper_limit

#remove
outliers_data = np.where(data['crim'] > upper_limit, True, np.where(data['crim'] < lower_limit, True, False))
data_trimmed = data.loc[~(outliers_data), ]
data.shape, data_trimmed.shape
sns.boxplot(data_trimmed.crim)

#replaced with upper or lower limit
data['crim_replaced'] = pd.DataFrame(np.where(data['crim'] > upper_limit, upper_limit, np.where(data['crim'] < lower_limit, lower_limit, data['crim'])))
sns.boxplot(data.crim_replaced)
#winsorizer
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',
                          tail='both', 
                          fold=1.5,
                          variables=['crim'])
df_t = winsor.fit_transform(data[['crim']])
sns.boxplot(df_t.crim)
####################zn

sns.boxplot(data.zn)
IQR = data['zn'].quantile(0.75) - data['zn'].quantile(0.25)
IQR
lower_limit = data['zn'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['zn'].quantile(0.75) + (IQR * 1.5)
upper_limit



outliers_data = np.where(data['zn'] > upper_limit, True, np.where(data['zn'] < lower_limit, True, False))
data_trimmed = data.loc[~(outliers_data), ]
data.shape, data_trimmed.shape
sns.boxplot(data_trimmed.crim)

data['zn_replaced'] = pd.DataFrame(np.where(data['zn'] > upper_limit, upper_limit, np.where(data['zn'] < lower_limit, lower_limit, data['zn'])))
sns.boxplot(data.zn_replaced)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['zn'])
df_t = winsor.fit_transform(data[['zn']])
sns.boxplot(df_t.zn)
##########################################
sns.boxplot(data.indus)  #no outlier
sns.boxplot(data.chas)    #no outlier
sns.boxplot(data.nox)      #no outlier

##################rm
sns.boxplot(data.rm)   #outlier both the side
IQR = data['rm'].quantile(0.75) - data['rm'].quantile(0.25)
IQR
lower_limit = data['rm'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['rm'].quantile(0.75) + (IQR * 1.5)
upper_limit

data['RM_replaced'] = pd.DataFrame(np.where(data['rm'] > upper_limit, upper_limit, np.where(data['rm'] < lower_limit, lower_limit, data['rm'])))
sns.boxplot(data.RM_replaced)

sns.boxplot(data.age)   #NO OUTLIER

#################dis
sns.boxplot(data.dis)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['dis'])
df_t = winsor.fit_transform(data[['dis']])
sns.boxplot(df_t.dis)


sns.boxplot(data.rad)
sns.boxplot(data.tax)

############ ptratio has a outlier
sns.boxplot(data.ptratio)
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['ptratio'])
df_t = winsor.fit_transform(data[['ptratio']])
sns.boxplot(df_t.ptratio)


############# black
sns.boxplot(data.black)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', 
                          tail='both',
                          fold=1.5,
                          variables=['black'])
df_t = winsor.fit_transform(data[['black']])
sns.boxplot(df_t.black)
############### lstat
sns.boxplot(data.lstat)

IQR = data['lstat'].quantile(0.75) - data['lstat'].quantile(0.25)
lower_limit = data['lstat'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['lstat'].quantile(0.75) + (IQR * 1.5)

data['lstat_replaced'] = pd.DataFrame(np.where(data['lstat'] > upper_limit, upper_limit, np.where(data['lstat'] < lower_limit, lower_limit, data['lstat'])))
sns.boxplot(data.lstat_replaced)

########### medv has outlier
sns.boxplot(data.medv)

IQR = data['medv'].quantile(0.75) - data['medv'].quantile(0.25)
IQR
lower_limit = data['medv'].quantile(0.25) - (IQR * 1.5)
lower_limit
upper_limit = data['medv'].quantile(0.75) + (IQR * 1.5)
upper_limit

data['medv_replaced'] = pd.DataFrame(np.where(data['medv'] > upper_limit, upper_limit, np.where(data['medv'] < lower_limit, lower_limit, data['medv'])))
sns.boxplot(data.medv_replaced)
