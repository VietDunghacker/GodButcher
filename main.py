import nltk
import random
import csv
import re
import string
import os
import sys
import utilities
import time
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

ps = PorterStemmer() #stemmer
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

#collect train data
collected_data = []
toxic_data = set()
clean_data = set()
train_data = []
test_data = []
with open("./toxic_comment/train.csv") as f:
	data = csv.reader(f)
	next(data)
	for row in data:
		collected_data.append((row[1],(int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]), int(row[7]))))
collected_data = list(set(collected_data))[:20000]
for (comment, tag) in collected_data:
	if sum(tag) == 0:
		clean_data.add((comment,tag))
	else:
		toxic_data.add((comment,tag))
toxic_data = list(toxic_data)
clean_data = list(clean_data)
train_data = toxic_data[int(0.1 * len(toxic_data)):] + clean_data[int(0.5 * len(clean_data)):]
test_data = toxic_data[:int(0.1 * len(toxic_data))] + clean_data[:int(0.5 * len(clean_data))]
'''tags = [0] * 7
for (comment, tag) in train_data:
	if sum(tag) == 0:
		tags[6] += 1
	else:
		for i in range(6):
			tags[i] += tag[i]
for i in range(len(tags)):
	print(tags[i])
'''
#negative features. Input is a sentence (raw string)
count = 0
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
	dic.update(utilities.dependency_features(sent))
	return dic

#classify data using NaiveBayes
feature_vector_train_data = [negative_features(sent) for (sent,tag) in train_data]
feature_vector_test_data = [negative_features(sent) for (sent,tag) in test_data]
for label in range(6):
	train_set, test_set = [], []
	for i in range(len(train_data)):
		train_set.append((feature_vector_train_data[i], train_data[i][1][label]))
	for i in range(len(test_data)):
		test_set.append((feature_vector_test_data[i], test_data[i][1][label]))
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

	print("Label {}: Accuracy = {} Precision = {} Recall = {}".format(labels[label], nltk.classify.accuracy(classifier, test_set), precision, recall))
	#classifier.show_most_informative_features(50)
	#print(classifier.pseudocode(depth = 50))
