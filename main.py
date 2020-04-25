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
def num_word(raw):
    '''
    count the number of word (duplicated)
    a word is defined as a substring seperated by \space
    input: raw : string
    out put: dict num word: number of word 
    '''
    return ({'num word': len(raw.split())})
def num_unique_word(raw):
    '''
    count number of unique word (not duplicated)
    input raw: string
    output: dict num_unique_word: number of unique word
    '''
    return ({'num unique word':len(set(raw.split()))})
def ration_unique(raw):
    '''
    compute ration of unique word
    a word is a substring seperated by \space
    input raw : str
    output: dict ration_unique : ration of unique word
    '''
    return ({'ration unique':num_unique_word(raw)['num unique word']/num_word(raw)['num word']})
def num_token_no_stop(tokened):
    '''
    return number of token without stop word (depend on token function, can conclude '.', ',')
    duplicated
    intput tokened list : list
    output dict num token no stop: number of tokens without stop word 
    '''
    stop= stopwords.words('english')
    no_stop= [w for w in tokened if w.lower() not in stop ]
    return ({'num token no stop':len(no_stop)})
def num_spelling_error(tokened):
    '''
    return number of word not in English vocab
    can be counted differences, women, measures (hàm kém vl)
    can be counted ',','.', 
    input tokened: list
    output: dict num spelling error :number of tokens not in English vocab
    '''
    words= nltk.corpus.words.words()
    spell_wrong= [w for w in tokened if w.lower() not in words]
    return ({'num spelling error':len(spell_wrong)})
def num_allcap(tokened):
    '''
    return number of word written all captial (duplicated)
    input tokened: list
    output: dict num all cap: number of tokens written all capital
    '''
    cap= [w for w in tokened if re.search('^[A-Z]+$', w)]
    return ({'num all cap':len(cap)})
def rate_allcap(raw, tokened):
    '''
    return portion of word written all capital (duplicated)
    input: raw: str
           tokened: list
    output: dict rate all cap :rate of all capital word
    '''
    r=num_allcap(tokened)['num all cap']/ num_word(raw)['num word']
    return ({'rate all cap':r})
def length_cmt(raw):
    '''
    return length of the cmt
    intput raw: str
    output: dict 'length cmt':length of the cmt
    '''
    return ({'length cmt': len(raw)})
def num_cap_letter(raw):
    '''
    return number of capital letter
    input raw: str
    output: dict 'num cap letter':number of capital letter
    '''
    cap= [w for w in raw if re.search('[A-Z]',w)]
    return({'num cap letter':len(cap)})
def rate_cap_letter(raw):
    '''
    return ratio of capital letter
    input raw: str
    outout: dict 'rate cap letter': ratio of capital letter
    '''
    r=num_cap_letter(raw)['num cap letter']/len(raw)
    return({'rate cap letter':r})
def num_explan_mark(raw):
    '''
    return number of explanation mark (not necessary using as explanation)
    input raw: str
    output: dict 'num explan mark': number of explanation mark
    '''
    count=0
    for c in raw:
        if c=='!':
            count=count+1
    return {'num explan mark':count}
def rate_explan_mark(raw):
    '''
    return rate of explanation mark (not necessary using as explanation)
    input raw: str
    output: rate of explanation mark
    '''
    r=num_explan_mark(raw)['num explan mark']/len(raw)
    return({'rate explan mark':r})
def num_quest_mark(raw):
    '''
    return number of question mark (not necessary using as question)
    input raw: str
    output: dict 'num quest mark' number of question mark
    '''
    count=0
    for c in raw:
        if c=='?':
            count=count+1
    return {'num quest mark' :count}
def rate_quest_mark(raw):
    '''
    return rate of question mark (not necessary using as question)
    input raw: str
    output: dict 'rate quest mark':rate of question mark
    '''
    r= num_quest_mark(raw)['num quest mark']/len(raw)
    return {'rate quest mark', r}
def num_punc_mark( raw):
    '''
    return number of punctuation mark (not necessary using as finish setences)
    input tokened: str
    output: dict 'num punc mark': number of punctuation mark
    '''
    count=0
    for c in raw:
        if c=='.':
            count=count+1
    return {'num ounc mark': count}
def num_mark_sym( raw):
    '''
    return number of marking symbol (*, &,$,%)
    input raw: str
    output: dict 'num mark sym' :number of marking symbol mark
    '''
    count=0
    for c in raw:
        if c in {'*','&','$','%'}:
            count=count+1
    return {'num mark sym': count}
def num_smile( tokened):
    '''
    Count the number of emoji (can not count the case hoang:) or :))))
    input: tokened: list
    output:dict 'num smile': number of smile
    '''
    count= 0
    for w in bigrams(tokened):
        if w== (':',')'):
            count=count+1
    return {'num smile':count}
def rate_lower(raw):
    '''
    Count the rate of lowercase character
    input raw: str
    output: dict 'rate lower': rate of lowercase letter
    '''
    l= [w for w in raw if re.search('[a-z]',w)]
    return({'rate lower':len(l)/len(raw)})
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