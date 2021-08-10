import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import pandas as pd
import numpy as np

df  = pd.read_csv(r"F:\data sci Assignment\12-NLP\New folder\twiter\twit.csv")
df.head()

#cleanning the data

df = df.drop(['id'], axis = 1)

df['tweet'][0]

df = df.iloc[:,[1,0]]
df.head()



df = df.sample(20000)
#shape
df.shape
df.info()

#now remove all the html tag in reviews col:
#use dif function to remove all tag
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean,'', text)

df['tweet'] = df['tweet'].apply(clean_html)

#converting everthing in lower case
def convert_text(text):
    return text.lower()

df['tweet'] = df['tweet'].apply(convert_text)

#remove 
def remove_nonalpha(text):    
    clean = re.compile("[^a-zA-Z]")
    return  re.sub(clean," ",text)

df['tweet'] = df['tweet'].apply(remove_nonalpha)


# reomve the stop word
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')
  
stopwords.words('english')

#use def funtion to remove stop words

def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y = x[:]
    x.clear()
    return y
        
df['tweet'] = df['tweet'].apply(remove_stopwords)


#now perform steaming

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

y = []
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z

df['tweet'] = df['tweet'].apply(stem_words)
#now all the prepocessing is done we have clean data
df


#join back
def join_back(list_input):
    return " ".join(list_input)

df['tweet'] = df['tweet'].apply(join_back)
df['tweet']



X = df.iloc[:,0:1].values

X.shape


# Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)

X = cv.fit_transform(df['tweet']).toarray()
X.shape

y = df.iloc[:,-1].values
y.shape

#now data is diveded ino 2 part 
#X = test set
#y = trainning set

from sklearn.model_selection import train_test_split
X_train, X_text, y_train, y_test=train_test_split(X,y,test_size=0.2)

X_train.shape
X_text.shape  


y_train.shape
y_test.shape  
# bulit three model
#GaussianNB
#,MultinomialNB
#BernoulliNB
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)

y_perd1 = clf1.predict(X_text)
y_perd2 = clf2.predict(X_text)
y_perd3 = clf3.predict(X_text)


y_test.shape
y_perd1.shape

#find acccuricy score to use accuracy_score

from sklearn.metrics import accuracy_score

print("GaussianNB",accuracy_score(y_test,y_perd1))
print("MultinomialNB",accuracy_score(y_test,y_perd2))
print("BernoulliNB",accuracy_score(y_test,y_perd3))












