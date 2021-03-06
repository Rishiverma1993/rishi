import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
oneplus_revies=[]

for i in range(1,21):
  ip=[]  
  url="https://www.amazon.in/OnePlus-Glacial-Green-128GB-Storage/product-reviews/B078BN2H3R/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  oneplus_revies=oneplus_revies+ip 
  
# writng reviews in a text file 
with open("oneplusG.txt","w",encoding='utf8') as output:
    output.write(str(oneplus_revies))  

  
ip_rev_string = " ".join(oneplus_revies)
import nltk

# from nltk.corpus import stopwords

#preprosccing the data remove all element except char:
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
#now tokonnizetion
ip_review_words = ip_rev_string.split(" ")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_review_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_review_words)

#import stop word
with open("F:/data sci Assignment/12-NLP/New folder/stop.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["oneplus","mobile","time","android","phone","device","screen","battery","product","good","day","price"])

ip_review_words = [w for w in ip_review_words if not w in stop_words]


# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_review_words)    

# WordCloud can be performed on the string inputs.
# Corpus level word cloud
#all corpus in fig 1:
wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

#Sentiment analysis (or opinion mining) is a natural language processing technique
# used to determine whether data is positive, negative 

#import pos dic 
with open("F:\\data sci Assignment\\12-NLP\\New folder\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_review_words if w in poswords])

#make pos word cloud in fig 2
wordcloud_pos_in_pos = WordCloud(
                      background_color='pink',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

#importing neg dic :
with open ("F:\\data sci Assignment\\12-NLP\\New folder\\negative-words.txt","r") as neg:
    negwords = neg.read().split("\n")    
# Positive word cloud
# Choosing the only words which are present in positive words

ip_neg_in_neg = " ".join ([w for w in ip_review_words if w in negwords])

#make a neg word cloud in fig 3
wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

#word cloud bigram
nltk.download('punkt')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!???????``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great','html','tests','contrast','devices'] 

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]



nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)


dict_2 = [' '.join(tup) for tup in bigrams_list]
print (dict_2)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dict_2)
vectorizer.vocabulary_

sum_wrd = bag_of_words.sum(axis=0)
words_freq = [(word, sum_wrd[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

#make a dic
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)


plt.figure(4)
plt.title('bigrams same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()







