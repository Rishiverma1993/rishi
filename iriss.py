# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 00:47:35 2021

@author: rishi
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("F:\data sci Assignment\data prepro csv file\iris.csv")
data
data['size'] = pd.cut(data['Sepal.Length'],3,labels=['low','medium','high'])
data.head(149)
