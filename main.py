import nltk
import random
import csv
import re
import string
import os
import sys
import time
import pandas as pd
import random
import markovify as mk
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

ps = PorterStemmer() #stemmer
stopwords = set(nltk.corpus.stopwords.words('english'))
englishwords = set(nltk.corpus.words.words())
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

list_features = ["num_word","num_unique_word", "rate_unique", "num_token_no_stop", "num_spelling_error", "num_all_cap", "rate_all_cap", "length_cmt", "num_cap_letter", "rate_cap_letter", "num_explan_mark", "rate_explan_mark","num_quest_mark", "rate_quest_mark","num_punc_mark","num_mark_sym","num_smile","rate_space","rate_lower","bad_words_type_1","bad_words_type_2","bad_words_type_3","bad_words_type_4","bad_words_type_5","bad_words_all_type", "sentimental_score"]
list_dependency_features = ["dependency_proper_noun_singular", "dependency_proper_noun_plural", "dependency_personal_pronoun", "dependency_possessive_pronoun", "dependency_with_denial", "dependency_denial_contain_proper_noun_singular", "dependency_denial_contain_proper_noun_plural", "dependency_denial_contain_personal_pronoun", "dependency_denial_contain_possessive_pronoun", "dependency_proper_noun_singular_and_denial", "dependency_proper_noun_plural_and_denial", "dependency_personal_pronoun_and_denial", "dependency_possessive_pronoun_and_denial", "dependency_contain_bad_words", "dependency_denial_contain_bad_words", "dependency_proper_noun_singular_bad_words", "dependency_proper_noun_plural_bad_words", "dependency_personal_pronoun_bad_words", "dependency_possessive_pronoun_bad_words", "dependency_pronoun_bad_words"]
list_features = list_features+ list_dependency_features
remove_list = ["dependency_denial_contain_proper_noun_plural", "dependency_proper_noun_plural_and_denial", "dependency_proper_noun_plural_bad_words", "dependency_proper_noun_plural", "dependency_denial_contain_personal_pronoun", "dependency_denial_contain_possessive_pronoun", "dependency_denial_contain_bad_words"]
list_features = ["rate_unique", "num_token_no_stop", "num_spelling_error", "num_all_cap", "rate_all_cap", "length_cmt", "num_cap_letter", "rate_cap_letter", "rate_explan_mark","rate_space","rate_lower","bad_words_type_1","bad_words_type_2","bad_words_type_3","bad_words_type_4","bad_words_type_5","bad_words_all_type", "sentimental_score", "dependency_proper_noun_singular", "dependency_contain_bad_words", "dependency_personal_pronoun_bad_words"]

def shuffle_function():
	return 0.69
def evaluate(classifier, test_set, note, label):
	errorPP, errorPN, errorNP, errorNN = 0,0,0,0
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
	precision = 0
	recall = 0
	fmeasure = 0
	if errorPP + errorPN != 0:
		precision = errorPP/(errorPP + errorPN)
	if errorPP + errorNP != 0:
		recall = errorPP/(errorPP + errorNP)
	if precision + recall != 0:
		fmeasure = 2 * precision * recall / (precision + recall)
	print("Note: {}.Label {}: Precision = {:.4f} Recall = {:.4f} F-measure = {:.4f}".format(note,label, precision, recall, fmeasure))

def bag_of_words(text):
	dic = {}
	words = nltk.word_tokenize(text)
	words = [word.lower() for word in words if not word in stopwords]
	for i, word in enumerate(words):
  		if word in englishwords:
  			words[i] = ps.stem(word)
	#tagged = nltk.pos_tag(words, tagset = 'universal')
	fdist = nltk.FreqDist(words)
	for key in fdist.keys():
		dic['contain_{}'.format(key)] = fdist[key]
	return dic

if not 'features.csv' in os.listdir():
	os.system('python feature_handler.py')

start = time.time()
#read features already collected
feature_data = pd.read_csv('./features.csv', header = 0)
collected_data = [] #a tuple(dictionary of features, tuple of six labels)

idnum_dict = {}
train_file = pd.read_csv('./toxic_comment/train.csv', header = 0)
for i in range(len(train_file)):
    row = train_file.iloc[i]
    idnum_dict[row['id']] = row['comment_text']

#collect data into collected_data
for i in range(len(feature_data)):
	dic = {}
	tags = []
	row = feature_data.iloc[i]
	if len(row) == 0:
		continue
	idnum = row['Comment']
	for feature in list_features:
		dic[feature] = float(row[feature])
	for label in labels:
		tags.append(int(row[label]))
	collected_data.append((idnum,dic,tuple(tags)))
print("Collect data time is {}".format(time.time() - start))
start = time.time()



#seperate collected_data into clean_data and toxic_data
clean_data = [] #data not labeled
toxic_data = [] #data labled at least one
for (idnum, features, tags) in collected_data:
  #features.update(bag_of_words(idnum_dict[idnum]))
  if sum(tags) == 0:
    clean_data.append((features,tags))
  else:
    toxic_data.append((features,tags))
random.shuffle(clean_data, shuffle_function)
random.shuffle(toxic_data, shuffle_function)
print(len(toxic_data))

#select train data and test data
train_size = int(0.2 * len(toxic_data))
train_data = toxic_data[train_size:]
test_data = toxic_data[:train_size]
train_data = train_data + clean_data[ : int(0.9 * len(clean_data))]
test_data = test_data + clean_data[int(0.9 * len(clean_data)) : ]
print("Classifying time is {}".format(time.time() - start))
start = time.time()

classifiers = [None] * 6
for i in range(6):
  train_set = [(feature, tags[i]) for (feature, tags) in train_data]
  test_set = [(feature, tags[i]) for (feature, tags) in test_data]
  print('finish collecting data')
  classifier = nltk.NaiveBayesClassifier.train(train_set) #choose Naive Bayes classification
  classifiers[i] = classifier
  evaluate(classifier, train_set, 'train set', labels[i])
  evaluate(classifier, test_set, 'test set', labels[i])

#classifier = nltk.DecisionTreeClassifier.train(train_data) #choose Decision Tree classification

#calculate precision and recall
#classifier.show_most_informative_features(30)
end  = time.time()
print("total time is {}".format(end - start))
'''while(True):
	sample = input("Please enter a string:")
	if sample == 'quit':
		break
	print(sample)
	featured_sample = feature_handler.negative_features(sample)
	for key in remove_list:
		del featured_sample[key]
	featured_sample = utilities.scale_sample(featured_sample, sample_mean_train_data, sample_dev_train_data)
	print(featured_sample)
	for label in range(6):
		res = classifiers[label].classify(featured_sample)
		print("label {}: {}".format(labels[label],res))'''
