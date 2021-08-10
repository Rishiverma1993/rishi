import pandas as pd # Read data from file 'filename.csv' , delimiters, rows, column names
import numpy as np   #used for working with arrays

claim = pd.read_csv(r"F:\data sci Assignment\New folder\claimants.csv")
claim


claim.isna().sum()

from sklearn.impute import SimpleImputer  #Imputation transformer for completing missing values.

# (Mean Imputer ). “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
claim_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

claim["CLMAGE"] = pd.DataFrame(claim_mean.fit_transform(claim[["CLMAGE"]]))

claim["CLMAGE"].isna().sum()
claim

claim.isna().sum()



#(median imputer)then replace missing values using the median along each column. Can only be used with numeric data.


claim_median = SimpleImputer(missing_values=np.nan, strategy='median')

claim["CLMAGE"] = pd.DataFrame(claim_mean.fit_transform(claim[["CLMAGE"]]))

claim["CLMAGE"].isna().sum()
claim

#both mean and median are  same:
#Mean and Median imputer are used for numeric data
    

#Mode is used for discrete data
#Mode imputation means replacing missing values by the mode, or the most frequent- category value. 
#The results of this imputation will look like this: It's good to know that the above imputation methods.   
claim_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

claim["SEATBELT"] = pd.DataFrame(claim_mode.fit_transform(claim[["SEATBELT"]]))

claim["CLMINSUR"] = pd.DataFrame(claim_mode.fit_transform(claim[["CLMINSUR"]]))

claim["CLMSEX"] = pd.DataFrame(claim_mode.fit_transform(claim[["CLMSEX"]]))

claim.isnull().sum()    
claim
