import csv
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.util import bigrams

stopwords = set(stopwords.words('english')) #stopword
englishwords = set(nltk.corpus.words.words())
bad_words = {}

for file in os.listdir("./bad_word/csv_file/"):
    with open("./bad_word/csv_file/" + file, errors = 'ignore') as f:
        data = csv.reader(f, delimiter = ";")
        temp_data = set()
        for row in data:
            if(len(row) > 0):
                temp_data.add(row[0])
        bad_words[file] = temp_data
for file in os.listdir("./bad_word/txt_file/"):
    with open("./bad_word/txt_file/" + file, errors = 'ignore') as f:
        temp_data = set()
        for row in f:
            if(len(row.strip()) > 0):
                temp_data.add(row.strip())
        bad_words[file] = temp_data
for file in os.listdir("./bad_word/comma_seperated_txt_file/"):
    with open("./bad_word/comma_seperated_txt_file/" + file, errors = 'ignore') as f:
        temp_data = set()
        for row in f:
            if not (len(row.strip()) == 0 or row.strip().startswith("##")):
                for word in row.strip().split(', '):
                    temp_data.add(word)
        bad_words[file] = temp_data

def num_word(raw):
    '''
    count the number of word (duplicated)
    a word is defined as a substring seperated by space
    input: raw : string
    out put: dict num word: number of word 
    '''
    return ({'num_word': len(raw.split())})
def num_unique_word(raw):
    '''
    count number of unique word (not duplicated)
    input raw: string
    output: dict num_unique_word: number of unique word
    '''
    return ({'num_unique_word':len(set(raw.split()))})
def ration_unique(raw):
    '''
    compute ration of unique word
    a word is a substring seperated by space
    input raw : str
    output: dict ration_unique : ration of unique word
    '''
    rate = num_unique_word(raw)['num_unique_word']/num_word(raw)['num_word']
    return ({'ration unique': round(rate,3)})
def num_token_no_stop(tokened):
    '''
    return number of token without stop word (depend on token function, can conclude '.', ',')
    duplicated
    intput tokened list : list
    output dict num token no stop: number of tokens without stop word 
    '''
    no_stop= [w for w in tokened if w.lower() not in stopwords]
    return ({'num_token_no_stop':len(no_stop)})
def num_spelling_error(tokened):
    '''
    return number of word not in English vocab
    can be counted differences, women, measures (hàm kém vl)
    can be counted ',','.', 
    input tokened: list
    output: dict num spelling error :number of tokens not in English vocab
    '''
    spell_wrong= [w for w in tokened if w.lower() not in englishwords]
    return ({'num_spelling_error':len(spell_wrong)})
def num_allcap(tokened):
    '''
    return number of word written all captial (duplicated)
    input tokened: list
    output: dict num all cap: number of tokens written all capital
    '''
    cap= [w for w in tokened if w.isupper()]
    return ({'num_all_cap':len(cap)})
def rate_allcap(raw, tokened):
    '''
    return portion of word written all capital (duplicated)
    input: raw: str
           tokened: list
    output: dict rate all cap :rate of all capital word
    '''
    rate = num_allcap(tokened)['num_all_cap']/ num_word(raw)['num_word']
    return ({'rate_all_cap': round(rate, 3)})
def length_cmt(raw):
    '''
    return length of the cmt
    intput raw: str
    output: dict 'length cmt':length of the cmt
    '''
    return ({'length_cmt': len(raw)})
def num_cap_letter(raw):
    '''
    return number of capital letter
    input raw: str
    output: dict 'num cap letter':number of capital letter
    '''
    cap = [w for w in raw if w.isupper()]
    return({'num_cap_letter':len(cap)})
def rate_cap_letter(raw):
    '''
    return ratio of capital letter
    input raw: str
    outout: dict 'rate cap letter': ratio of capital letter
    '''
    rate = num_cap_letter(raw)['num_cap_letter']/len(raw)
    return({'rate_cap_letter':round(rate, 3)})
def num_explan_mark(raw):
    '''
    return number of explanation mark (not necessary using as explanation)
    input raw: str
    output: dict 'num explan mark': number of explanation mark
    '''
    count=0
    for c in raw:
        if c=='!':
            count = count+1
    return {'num_explan_mark':count}
def rate_explan_mark(raw):
    '''
    return rate of explanation mark (not necessary using as explanation)
    input raw: str
    output: rate of explanation mark
    '''
    rate = num_explan_mark(raw)['num_explan_mark']/len(raw)
    return({'rate_explan_mark':round(rate, 3)})
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
    return {'num_quest_mark' : count}
def rate_quest_mark(raw):
    '''
    return rate of question mark (not necessary using as question)
    input raw: str
    output: dict 'rate quest mark':rate of question mark
    '''
    rate = num_quest_mark(raw)['num_quest_mark']/len(raw)
    return {'rate quest mark' : round(rate, 3)}
def num_punc_mark(raw):
    '''
    return number of punctuation mark (not necessary using as finish setences)
    input tokened: str
    output: dict 'num punc mark': number of punctuation mark
    '''
    count=0
    for c in raw:
        if c=='.':
            count=count+1
    return {'num_ounc_mark': count}
def num_mark_sym(raw):
    '''
    return number of marking symbol (*, &,$,%)
    input raw: str
    output: dict 'num mark sym' :number of marking symbol mark
    '''
    count=0
    for c in raw:
        if c in {'*','&','$','%'}:
            count=count+1
    return {'num_mark_sym': count}
def num_smile(tokened):
    '''
    Count the number of emoji (can not count the case hoang:) or :))))
    input: tokened: list
    output:dict 'num smile': number of smile
    '''
    count= 0
    for w in bigrams(tokened):
        if w== (':',')'):
            count=count+1
    return {'num_smile':count}
def rate_lower(raw):
    '''
    Count the rate of lowercase character
    input raw: str
    output: dict 'rate lower': rate of lowercase letter
    '''
    l = [w for w in raw if w.islower()]
    rate = len(l)/len(raw)
    return({'rate_lower': round(rate, 3)})
# input: list of word, output: integer
def x20(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w in bad_words['bad-words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_1"] = count
    return (dic)
# input: list of word, output: integer
def x21(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w in bad_words['swearWords.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_2"] = count
    return (dic)
# input: list of word, output: integer
def x22(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w in bad_words['facebook_bad_words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_3"] = count
    return (dic)

# input: list of words, output: integer
def x23(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w in bad_words['youtube_bad_words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_4"] = count
    return (dic)

#input: list of words, output:integer
def x24(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w in bad_words['google_twunter_lol.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_5"] = count
    return (dic)

#input: list of words, output:integer
def x25(tokened):
    distinct_bad_words = set()
    dic = {}
    for key in bad_words.keys():
        distinct_bad_words = distinct_bad_words.union(bad_words[key])
    count=0
    dic = {}
    for w in tokened:
        if w in distinct_bad_words:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_all_type"] = count
    return (dic)