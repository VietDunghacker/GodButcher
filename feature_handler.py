import nltk
import random
import csv
import re
import string
import os
import sys
import utilities
import time
import multiprocessing 
import urllib.request
import zipfile
from urllib import request
from nltk import FreqDist
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.parse.corenlp import CoreNLPServer

if not "stanford-corenlp-4.0.0" in os.listdir():
	urllib.request.urlretrieve('http://nlp.stanford.edu/software/stanford-corenlp-latest.zip', 'stanford-corenlp-latest.zip')
	zipfile.ZipFile('stanford-corenlp-latest.zip', 'r').extractall('./')

STANFORD = "./stanford-corenlp-4.0.0"
server = CoreNLPServer("./stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar", "./stanford-corenlp-4.0.0/stanford-corenlp-4.0.0-models.jar",)


initialized = False #if this is false, the initialize() function will be running
up_to_date = True #check if this is up-to-date

ps = PorterStemmer() #stemmer
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
list_normal_features = ["num_word","num_unique_word", "rate_unique", "num_token_no_stop", "num_spelling_error", "num_all_cap", "rate_all_cap", "length_cmt", "num_cap_letter", "rate_cap_letter", "num_explan_mark", "rate_explan_mark","num_quest_mark", "rate_quest_mark","num_punc_mark","num_mark_sym","num_smile","rate_space","rate_lower","bad_words_type_1","bad_words_type_2","bad_words_type_3","bad_words_type_4","bad_words_type_5","bad_words_all_type"]
list_sentiment_features = ["sentimental_score"]
list_dependency_features = ["dependency_proper_noun_singular", "dependency_proper_noun_plural", "dependency_personal_pronoun", "dependency_possessive_pronoun", "dependency_with_denial", "dependency_denial_contain_proper_noun_singular", "dependency_denial_contain_proper_noun_plural", "dependency_denial_contain_personal_pronoun", "dependency_denial_contain_possessive_pronoun", "dependency_proper_noun_singular_and_denial", "dependency_proper_noun_plural_and_denial", "dependency_personal_pronoun_and_denial", "dependency_possessive_pronoun_and_denial", "dependency_contain_bad_words", "dependency_denial_contain_bad_words", "dependency_proper_noun_singular_bad_words", "dependency_proper_noun_plural_bad_words", "dependency_personal_pronoun_bad_words", "dependency_possessive_pronoun_bad_words", "dependency_pronoun_bad_words"]
list_features = list_normal_features + list_sentiment_features + list_dependency_features
#collect train data
collected_data = []


def negative_features(sent):
	words = word_tokenize(sent) #tokenize into list of words
	#clean the data
	#tag_words = nltk.pos_tag(words) #add tag into each word
	#add features
	dic = {}
	dic.update(utilities.num_word(sent))
	dic.update(utilities.num_unique_word(sent))
	dic.update(utilities.rate_unique(sent))
	dic.update(utilities.num_token_no_stop(words))
	dic.update(utilities.num_spelling_error(words))
	dic.update(utilities.num_all_cap(words))
	dic.update(utilities.rate_all_cap(sent,words))
	dic.update(utilities.length_cmt(sent))
	dic.update(utilities.num_cap_letter(sent))
	dic.update(utilities.rate_cap_letter(sent))
	dic.update(utilities.num_explan_mark(sent))
	dic.update(utilities.rate_explan_mark(sent))
	dic.update(utilities.num_quest_mark(sent))
	dic.update(utilities.rate_quest_mark(sent))
	dic.update(utilities.num_punc_mark(sent))
	dic.update(utilities.num_mark_sym(sent))
	dic.update(utilities.rate_space(sent))
	dic.update(utilities.num_smile(words))
	dic.update(utilities.rate_lower(sent))
	dic.update(utilities.x20(words))
	dic.update(utilities.x21(words))
	dic.update(utilities.x22(words))
	dic.update(utilities.x23(words))
	dic.update(utilities.x24(words))
	dic.update(utilities.x25(words))
	dic.update(utilities.stm_score(sent))
	dic.update(utilities.dependency_features(sent))
	return dic

#initialize the data
#processes: number of processes running. If leave it alone the program will consume all available resources
#a and b: interval of collected_data
def initialize(processes = 0, a = 0, b = 0):
	start = time.time()
	if initialized == False and a <= b:
		open_test_data()
		if b == 0:
			b = len(collected_data)
		times = int((len(collected_data[a : b]) - 1)/10000 + 1)
		with open('./features.csv', mode = 'w') as output:
			writer = csv.writer(output, delimiter = ',')
			writer.writerow(['Comment'] + labels + list_features)
			for i in range(times):
				server.start()
				if processes != 0:
					p = multiprocessing.Pool(processes)
				else:
					p = multiprocessing.Pool()
				results = p.map(extracting_features, collected_data[:10])
				for result in results:
					writer.writerow(result)
				server.stop()
	end = time.time()
	print(end-start)

def extracting_features(sentence):
	idnum, comment, tags = sentence
	result_dict = negative_features(comment)
	result_dict.update(tags)
	result_list = []
	for label in labels:
		result_list.append(result_dict[label])
	for feature in list_features:
		result_list.append(result_dict[feature])
	return [idnum] + result_list
#open the file containing train data
def open_test_data():
	with open("./toxic_comment/train.csv", encoding = 'utf-8') as f:
		data = csv.reader(f)
		next(data)
		for row in data:
			collected_data.append((row[0],row[1],{"toxic":int(row[2]),"severe_toxic":int(row[3]),"obscene":int(row[4]),"threat":int(row[5]),"insult":int(row[6]), "identity_hate":int(row[7])}))

#update new features
if __name__ == "__main__":
	initialize()