import nltk
import random
import csv
import re
import string
import os
import sys
from urllib import request
from nltk import FreqDist
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

#collect bad words
bad_words = set()

for file in os.listdir("./bad_word/csv_file/"):
	with open("./bad_word/csv_file/" + file, errors = 'ignore') as f:
		data = csv.reader(f, delimiter = ";")
		for row in data:
			if(len(row) > 0):
				bad_words.add(row[0])
for file in os.listdir("./bad_word/txt_file/"):
	with open("./bad_word/txt_file/" + file, errors = 'ignore') as f:
		for row in f:
			if(len(row.strip()) > 0):
				bad_words.add(row.strip())
for file in os.listdir("./bad_word/comma_seperated_txt_file/"):
	with open("./bad_word/comma_seperated_txt_file/" + file, errors = 'ignore') as f:
		for row in f:
			if not (len(row.strip()) == 0 or row.strip().startswith("##")):
				for word in word_tokenize(row.strip()):
					bad_words.add(word)

ps = PorterStemmer() #stemmer
stopwords = set(stopwords.words('english')) #stopword

#collect train data
train_data = []
with open("./toxic_comment/train.csv") as f:
	data = csv.reader(f)
	next(data)
	for row in data:
		train_data.append((row[1], {"toxic" : row[2], "severe_toxic" : row[3], "obscene" : row[4], "threat" : row[5], "insult": row[6], "identity_hate" : row[7]}))

def negative_features(sent):
	words = word_tokenize(sent)
	#clean the data

	#add features
	dic = {}
	return dic

#classify data using NaiveBayes
featuresets = [(negative_features(sent), tag) for (sent, tag) in train_data] #feature sets
size = int(0.1*len(featuresets))
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_data)

#calculate precision and recall
accuracy = nltk.classify.accuracy(classifier, test_data)
precision = 0
recall = 0
print("Accuracy = {} Precision = {} Recall = {}".format(accuracy, precision,recall))