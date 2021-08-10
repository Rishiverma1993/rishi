import pandas as pd

data = pd.read_excel("F:\data sci Assignment\Assignment_module02 (1).xlsx")
data


data.Points.mean() # '.' is used to refer to the variables within object
data.Points.median()
data.Points.mode()
data.Points.std() # standard deviation
range = max(data.Points) - min(data.Points) # range
range


data.Score.mean() # '.' is used to refer to the variables within object
data.Score.median()
data.Score.mode()
data.Score.std() # standard deviation
range = max(data.Score) - min(data.Score) # range
range



data.Weigh.mean() # '.' is used to refer to the variables within object
data.Weigh.median()
data.Weigh.mode()
data.Weigh.std() # standard deviation
range = max(data.Weigh) - min(data.Weigh) # range
range
