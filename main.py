import nltk
import random
import csv
import re
import string
import os
import sys
import utilities
import time
import pandas as pd
import random
import feature_handler.py
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

ps = PorterStemmer() #stemmer
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

list_features = ["num_word","num_unique_word", "rate_unique", "num_token_no_stop", "num_spelling_error", "num_all_cap", "rate_all_cap", "length_cmt", "num_cap_letter", "rate_cap_letter", "num_explan_mark", "rate_explan_mark","num_quest_mark", "rate_quest_mark","num_punc_mark","num_mark_sym","num_smile","rate_space","rate_lower","bad_words_type_1","bad_words_type_2","bad_words_type_3","bad_words_type_4","bad_words_type_5","bad_words_all_type", "sentimental_score"]
#start = time.time()

#read features already collected
feature_data = pd.read_csv('./features.csv', header = 0)
collected_data = [] #a tuple(dictionary of features, tuple of six labels)
clean_data = [] #data not labeled
toxic_data = [] #data labled at least one

#collect data into collected_data
for i in range(len(feature_data)):
	dic = {}
	tags = []
	row = feature_data.iloc[i]
	for feature in list_features:
		dic[feature] = float(row[feature])
	for label in labels:
		tags.append(int(row[label]))
	collected_data.append((dic,tuple(tags)))
#print("Collect data time is {}".format(time.time() - start))

#seperate collected_data into clean_data and toxic_data
for (features, tags) in collected_data:
	if sum(tags) == 0:
		clean_data.append((features,tags))
	else:
		toxic_data.append((features,tags))
random.shuffle(clean_data)
random.shuffle(toxic_data)

#select train data and test data
train_data = toxic_data[int(0.25 * len(toxic_data)):]
test_data = toxic_data[:int(0.25 * len(toxic_data))]
train_data = train_data + clean_data[int(0.25 * len(clean_data)):]
test_data = test_data + clean_data[:int(0.25 * len(clean_data))]

'''extract the feature into feature vectors and then scale the feature.
Each feature now has one of these three values: high, normal or medium, based on the normal distribution of 99% confidence interval'''
feature_vector_train_data = [feature for (feature, tag) in train_data]
feature_vector_test_data = [feature for (feature, tag) in test_data]
feature_vector_train_data = utilities.feature_scaling(feature_vector_train_data)
feature_vector_test_data = utilities.feature_scaling(feature_vector_test_data)

classifiers = [None] * 6 #contain 6 classifiers for each label
for label in range(6):
	train_set, test_set = [], []
	for i in range(len(train_data)):
		train_set.append((feature_vector_train_data[i], train_data[i][1][label]))
	for i in range(len(test_data)):
		test_set.append((feature_vector_test_data[i], test_data[i][1][label]))

	classifier = nltk.NaiveBayesClassifier.train(train_set) #choose Naive Bayes classification
	#classifier = nltk.DecisionTreeClassifier.train(train_set) #choose Decision Tree classification
	classifiers[label] = classifier

	#calculate precision and recall
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
	if(errorPP + errorPN == 0):
		precision = 0
	else:
		precision = round(errorPP/(errorPP + errorPN),4)
	if((errorPP + errorNP) == 0):
		recall = 0
	else:
		recall = round(errorPP/(errorPP + errorNP),4)

	print("Label {}: Accuracy = {} Precision = {} Recall = {}".format(labels[label], round(nltk.classify.accuracy(classifier, test_set),4), precision, recall))
	#classifier.show_most_informative_features(30) #print most informative features in Naive Bayes classification
	#print(classifier.pseudocode(depth = 20)) #print the pseudocode of Decision Tree classification

'''end  = time.time()
print("total time is {}".format(end - start))'''