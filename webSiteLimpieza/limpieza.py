# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:31:45 2020

@author: alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('csv_websites.csv', sep=',', index_col=0)

dataset.head(3)

print(dataset['post_text'].isnull().sum())

import re         
import nltk       
nltk.download('stopwords')

from nltk.corpus import stopwords

stopw = stopwords.words('spanish')

from nltk.stem.porter import PorterStemmer 

corpus = []                    

#stop_words_sp = set(stopwords.words(spanishstop_words_en = set(stopwords.words('english'))))
 
#stop_words = stop_words_sp | stop_words_en

for i in range(0, 6955):      
    review = re.sub('[^a-zA-Z]', ' ', dataset['title'].iloc[i]) 
    review = review.lower()         
    review = review.split()     
    ps = PorterStemmer()        
    review = [ps.stem(word) for word in review if not word in set(stopw)]  
    review = ' '.join(review)
    print(review)
    corpus.append(review) 
    
    
from sklearn.feature_extraction.text import CountVectorizer # 
cv = CountVectorizer(max_features = 1500)                   #

X = cv.fit_transform(corpus).toarray()                      #
y = dataset.iloc[:, 1].values                               #

