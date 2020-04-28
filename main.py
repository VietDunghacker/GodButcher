import nltk
import random
import csv
import re
import string
import os
import sys
import utilities
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

ps = PorterStemmer() #stemmer
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

#collect train data
train_data = []
with open("./toxic_comment/train.csv") as f:
	data = csv.reader(f)
	next(data)
	for row in data:
		train_data.append((row[1],{"toxic" : int(row[2]), "severe_toxic" : int(row[3]), "obscene" : int(row[4]), "threat" : int(row[5]), "insult": int(row[6]), "identity_hate" : int(row[7])}))

#negative features. Input is a sentence (raw string)
def negative_features(sent):
	words = word_tokenize(sent) #tokenize into list of words
	#clean the data
	#tag_words = nltk.pos_tag(words) #add tag into each word
	#add features
	dic = {}
	dic.update(utilities.num_word(sent))
	dic.update(utilities.num_unique_word(sent))
	dic.update(utilities.ration_unique(sent))
	dic.update(utilities.num_token_no_stop(words))
	dic.update(utilities.num_spelling_error(words))
	dic.update(utilities.num_allcap(words))
	dic.update(utilities.rate_allcap(sent,words))
	dic.update(utilities.length_cmt(sent))
	dic.update(utilities.num_cap_letter(sent))
	dic.update(utilities.rate_cap_letter(sent))
	dic.update(utilities.num_explan_mark(sent))
	dic.update(utilities.rate_explan_mark(sent))
	dic.update(utilities.num_quest_mark(sent))
	dic.update(utilities.rate_quest_mark(sent))
	dic.update(utilities.num_punc_mark(sent))
	dic.update(utilities.num_mark_sym(sent))
	dic.update(utilities.num_smile(words))
	dic.update(utilities.rate_lower(sent))
	dic.update(utilities.x20(words))
	dic.update(utilities.x21(words))
	dic.update(utilities.x22(words))
	dic.update(utilities.x23(words))
	dic.update(utilities.x24(words))
	dic.update(utilities.x25(words))
	return dic

#classify data using NaiveBayes
feature_vector = [negative_features(sent) for (sent,tag) in train_data]
for label in labels:
	featuresets = []
	for i in range(len(feature_vector)):
		featuresets.append((feature_vector[i], train_data[i][1][label]))
	size = int(0.1*len(featuresets))
	train_set, test_set = featuresets[size:], featuresets[:size]
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	errorPP, errorPN, errorNP, errorNN = 0, 0, 0, 0
	for (feature, tag) in test_set:
		predict = classifier.classify(feature)
		if predict == tag:
			if tag == 1:
				errorPP += 1
			else:
				errorNN += 1
		else:
			if tag == 1:
				errorNP += 1
			else:
				errorPN += 1
	precision = round(errorPP/(errorPP + errorPN),4)
	recall = round(errorNN/(errorNN + errorNP),4)

	print("Label {}: Accuracy = {} Precision = {} Recall = {}".format(label, nltk.classify.accuracy(classifier, test_set), precision, recall))
	#classifier.show_most_informative_features(50)
