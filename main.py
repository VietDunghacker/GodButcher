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
import sklearn
import utilities
import pickle
import feature_handler
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

ps = PorterStemmer() #stemmer
stopwords = set(nltk.corpus.stopwords.words('english'))
englishwords = set(nltk.corpus.words.words())
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

list_features = ["rate_unique", "num_token_no_stop", "num_spelling_error", "num_all_cap", "rate_all_cap", "length_cmt", "num_cap_letter", "rate_cap_letter", "rate_explan_mark","rate_space","rate_lower","bad_words_type_1","bad_words_type_2","bad_words_type_3","bad_words_type_4","bad_words_type_5","bad_words_all_type", "sentimental_score", "dependency_proper_noun_singular", "dependency_contain_bad_words", "dependency_personal_pronoun_bad_words"]

classifiers = pickle.load(open('store_classifier.txt','rb'))
feature_handler.server.start()
while(True):
	sample = input()
	if sample == 'quit':
		break
	if sample == '':
		continue
	featured_sample = feature_handler.negative_features(sample)
	toxics = []
	for key in list(featured_sample.keys()):
		if not key in list_features:
			del featured_sample[key]
	for label in range(6):
		res = classifiers[label].classify(featured_sample)
		if res == 1:
			toxics.append(labels[label])
	if len(toxics) > 0:
		print("Warning. Your comment is considered to be {}".format((", ").join(toxics)))
feature_handler.server.stop()