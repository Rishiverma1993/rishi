import pandas as pd

#####hot coding

data = pd.read_csv(r"F:\data sci Assignment\data prepro csv file\animal.csv")
data



pd.get_dummies(data['Animals']).head()
pd.get_dummies(data['Gender']).head()
pd.get_dummies(data['Homly']).head()
pd.get_dummies(data['Types']).head()


dummy_data = pd.get_dummies(data)
dummy_data

data_dummy = pd.get_dummies(data, columns=['Animals','Gender','Homly','Types'],drop_first=(True))
