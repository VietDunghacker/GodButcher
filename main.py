import nltk
import random
import csv
import re
import string
from urllib import request
from nltk import FreqDist
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

#collect bad words
url = "http://www.cs.cmu.edu/~biglou/resources/bad-words.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
raw = re.sub('\s+',' ',raw)
bad_words = raw.split()

ps = PorterStemmer() #stemmer
stopwords = set(stopwords.words('english')) #stopword

#collect train data
train_set = []

def negative_features(sent):
	words = word_tokenize(sent)
	#clean the data

	#add features
	dic = {}
	return dic

#classify data using NaiveBayes
featuresets = [] #feature sets
size = int(0.1*len(featuresets))
train_data, test_data = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_data)

#calculate precision and recall
precision = 0
recall = 0
print("Accuracy = {} Precision = {} Recall = {}".format(nltk.classify.accuracy(classifier, test_data), precision,recall))