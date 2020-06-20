import csv
import os
import sys
import nltk
import time
import numpy as np
import scipy.stats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.parse.corenlp import CoreNLPDependencyParser

high_threshold = scipy.stats.norm.ppf(.995,0,1)
low_threshold = scipy.stats.norm.ppf(.005,0,1)

ps = nltk.stem.PorterStemmer() #stemmer
dependency_parser = CoreNLPDependencyParser()
sid = SentimentIntensityAnalyzer() # One-time initialization

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
distinct_bad_words = set()
for key in bad_words.keys():
    distinct_bad_words = distinct_bad_words.union(bad_words[key])

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
def rate_unique(raw):
    '''
    compute ration of unique word
    a word is a substring seperated by space
    input raw : str
    output: dict ration_unique : ration of unique word
    '''
    rate = num_unique_word(raw)['num_unique_word']/num_word(raw)['num_word']
    return ({'rate_unique': rate})
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
def num_all_cap(tokened):
    '''
    return number of word written all captial (duplicated)
    input tokened: list
    output: dict num all cap: number of tokens written all capital
    '''
    cap= [w for w in tokened if w.isupper()]
    return ({'num_all_cap':len(cap)})
def rate_all_cap(raw, tokened):
    '''
    return portion of word written all capital (duplicated)
    input: raw: str
           tokened: list
    output: dict rate all cap :rate of all capital word
    '''
    rate = num_all_cap(tokened)['num_all_cap']/ num_word(raw)['num_word']
    return ({'rate_all_cap': rate})
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
    return({'rate_cap_letter':rate})
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
    return({'rate_explan_mark':rate})
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
    return {'rate_quest_mark' : rate}
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
    return {'num_punc_mark': count}
def num_mark_sym(raw):
    '''
    return number of marking symbol (*, &,$,%)
    input raw: str
    output: dict 'num mark sym' :number of marking symbol mark
    '''
    count=0
    for c in raw:
        if c in ['*','&','$','%']:
            count=count+1
    return {'num_mark_sym': count}
def num_smile(tokened):
    '''
    Count the number of emoji (can not count the case hoang:) or :))))
    input: tokened: list
    output:dict 'num_smile': number of smile
    '''
    count= 0
    for w in bigrams(tokened):
        if w== (':',')'):
            count=count+1
    return {'num_smile':count}
def rate_space(raw):
    '''
    Return the ratio of space
    input: raw string
    output:dict 'rate_space': ratio of space
    '''
    count = 0
    for c in raw:
        if c == " ":
            count += 1
    rate = count/len(raw)
    return {"rate_space": rate}
def rate_lower(raw):
    '''
    Count the rate of lowercase character
    input raw: str
    output: dict 'rate lower': rate of lowercase letter
    '''
    l = [w for w in raw if w.islower()]
    rate = len(l)/len(raw)
    return({'rate_lower': rate})
# input: list of word, output: integer
def x20(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w.lower() in bad_words['bad-words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_1"] = count
    return (dic)
# input: list of word, output: integer
def x21(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w.lower() in bad_words['swearWords.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_2"] = count
    return (dic)
# input: list of word, output: integer
def x22(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w.lower() in bad_words['facebook_bad_words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_3"] = count
    return (dic)

# input: list of words, output: integer
def x23(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w.lower() in bad_words['youtube_bad_words.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_4"] = count
    return (dic)

#input: list of words, output:integer
def x24(tokened):
    count=0
    dic = {}
    for w in tokened:
        if w.lower() in bad_words['google_twunter_lol.txt']:
            count+=1
            #dic["contains_{}".format(w)] = True
    dic["bad_words_type_5"] = count
    return (dic)

#input: list of words, output:integer
def x25(tokened):
    dic = {}
    count=0
    for w in tokened:
        if w.lower() in distinct_bad_words:
            count+=1
            #dic["contains_{}".format(w)] = 1
    dic["bad_words_all_type"] = count
    return (dic)

#################################################################################################
"""
from function x26 to x38, input is a raw string, for example "hello everyone", output is integer
"""
# count number of dependencies with proper nouns in the singular
def dependency_features(sent):
    dic = {}
    list_dependencies = set()
    sentences = nltk.tokenize.sent_tokenize(sent)

    result = dependency_parser.raw_parse_sents(sentences)
    for sentence in result:
        for item in next(sentence).triples():
            list_dependencies.add(item)

    list_function = [x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45]
    for function in list_function:
        dic.update(function(list_dependencies))
    return dic

def x26(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        type_words= [dependency[0][1],dependency[2][1]]
        if 'NNP' in type_words:
            count+=1
    dic = {}
    dic["dependency_proper_noun_singular"] = count
    return dic

# count number of dependencies with proper nouns in the plural
def x27(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        type_words= [dependency[0][1],dependency[2][1]]
        if 'NNP$' in type_words:
            count+=1
    dic = {}
    dic["dependency_proper_noun_plural"] = count
    return dic

# count number of dependencies with personal pronouns
def x28(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        type_words= [dependency[0][1],dependency[2][1]]
        if 'PRP' in type_words:
            count+=1
    dic = {}
    dic["dependency_personal_pronoun"] = count
    return dic

# count number of dependencies with possessive pronoun
def x29(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        type_words= [dependency[0][1],dependency[2][1]]
        if 'PRP$' in type_words:
            count+=1
    dic = {}
    dic["dependency_possessive_pronoun"] = count
    return dic

# count number of dependencies with denial (with words never or not)
def x30(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        words=[dependency[0][0],dependency[2][0]]
        if 'not' in words or 'never' in words:
            count+=1
    dic = {}
    dic["dependency_with_denial"] = count
    return dic

# count number of dependencies with denial that contain proper nouns in the singular
def x31(list_dependencies):
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][0] in ['not','never']) and (dependency[2][1]=='NNP')) or ((dependency[2][0] in ['not','never']) and (dependency[0][1]=='NNP')):
            count+=1
    dic = {}
    dic["dependency_denial_contain_proper_noun_singular"] = count
    return dic

#  number of dependencies with denial that contain proper nouns in the plural
def x32(list_dependencies):
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][0] in ['not','never']) and (dependency[2][1]=='NNP$')) or ((dependency[2][0] in ['not','never']) and (dependency[0][1]=='NNP$')):
            count+=1
    dic = {}
    dic["dependency_denial_contain_proper_noun_plural"] = count
    return dic

# number of dependencies with denial that contain personal pronouns
def x33(list_dependencies):
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][0] in ['not','never']) and (dependency[2][1]=='PRP')) or ((dependency[2][0] in ['not','never']) and (dependency[0][1]=='PRP')):
            count+=1
    dic = {}
    dic["dependency_denial_contain_personal_pronoun"] = count
    return dic

#count number of dependencies with denial that contain possessive pronouns
def x34(list_dependencies):
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][0] in ['not','never']) and (dependency[2][1]=='PRP$')) or ((dependency[2][0] in ['not','never']) and (dependency[0][1]=='PRP$')):
            count+=1
    dic = {}
    dic["dependency_denial_contain_possessive_pronoun"] = count
    return dic

# count number of dependencies between proper nouns in the singular and
# the words from dependencies with denial
def x35(list_dependencies):
    list_words_in_dependencies_with_denial0 = []
    for dependency in list_dependencies:
        words=[dependency[0][0],dependency[2][0]]
        if 'not' in words or 'never' in words:
            list_words_in_dependencies_with_denial0.extend(words)
    list_words_in_dependencies_with_denial1 = [ w for w in list_words_in_dependencies_with_denial0 if w not in ['not','never']]
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][1]== 'NNP') and (dependency[2][0] in list_words_in_dependencies_with_denial1)) or ((dependency[2][1]== 'NNP') and (dependency[0][0] in list_words_in_dependencies_with_denial1)):
            count+=1
    dic = {}
    dic["dependency_proper_noun_singular_and_denial"] = count
    return dic

# count number of dependencies between proper nouns in the plural
# and the words from dependencies with denial
def x36(list_dependencies):
    list_words_in_dependencies_with_denial0 = []
    for dependency in list_dependencies:
        words=[dependency[0][0],dependency[2][0]]
        if 'not' in words or 'never' in words:
            list_words_in_dependencies_with_denial0.extend(words)
    list_words_in_dependencies_with_denial1 = [ w for w in list_words_in_dependencies_with_denial0 if w not in ['not','never']]
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][1]== 'NNP$') and (dependency[2][0] in list_words_in_dependencies_with_denial1)) or ((dependency[2][1]== 'NNP$') and (dependency[0][0] in list_words_in_dependencies_with_denial1)):
            count+=1
    dic = {}
    dic["dependency_proper_noun_plural_and_denial"] = count
    return dic

# count number of dependencies between personal pronouns and the
# words from dependencies with denial
def x37(list_dependencies):
    list_words_in_dependencies_with_denial0 = []
    for dependency in list_dependencies:
        words=[dependency[0][0],dependency[2][0]]
        if 'not' in words or 'never' in words:
            list_words_in_dependencies_with_denial0.extend(words)
    list_words_in_dependencies_with_denial1 = [ w for w in list_words_in_dependencies_with_denial0 if w not in ['not','never']]
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][1]== 'PRP') and (dependency[2][0] in list_words_in_dependencies_with_denial1)) or ((dependency[2][1]== 'PRP') and (dependency[0][0] in list_words_in_dependencies_with_denial1)):
            count+=1
    dic = {}
    dic["dependency_personal_pronoun_and_denial"] = count
    return dic

# number of dependencies between possessive pronouns and the words
# from dependencies with denial
def x38(list_dependencies):
    list_words_in_dependencies_with_denial0 = []
    for dependency in list_dependencies:
        words=[dependency[0][0],dependency[2][0]]
        if 'not' in words or 'never' in words:
            list_words_in_dependencies_with_denial0.extend(words)
    list_words_in_dependencies_with_denial1 = [ w for w in list_words_in_dependencies_with_denial0 if w not in ['not','never']]
    # start to count
    count = 0
    for dependency in list_dependencies:
        if ((dependency[0][1]== 'PRP$') and (dependency[2][0] in list_words_in_dependencies_with_denial1)) or ((dependency[2][1]== 'PRP$') and (dependency[0][0] in list_words_in_dependencies_with_denial1)):
            count+=1
    dic = {}
    dic["dependency_possessive_pronoun_and_denial"] = count
    return dic

"""
from function x39 to x45, input include: list_bad_words is a list of bad words, 
sent_string is raw string
"""
def x39(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if (dependency[0][0] in distinct_bad_words) or (dependency[2][0] in distinct_bad_words):
            count+=1
    dic = {}
    dic["dependency_contain_bad_words"] = count
    return dic
# count number of dependencies with denial that contain the bad words
def x40(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if ((dependency[0][0] in distinct_bad_words) and (dependency[2][0] in ['not','never'])) or ((dependency[2][0] in distinct_bad_words) and (dependency[0][0] in ['not','never'])):
            count+=1
    dic = {}
    dic["dependency_denial_contain_bad_words"] = count
    return dic

# count number of dependencies between proper nouns in the singular and the bad words
def x41(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if ((dependency[0][1]=='NNP') and (dependency[2][0] in distinct_bad_words)) or ((dependency[2][1]=='NNP') and (dependency[0][0] in distinct_bad_words)):
            count+=1
    dic = {}
    dic["dependency_proper_noun_singular_bad_words"] = count
    return dic

# count number of dependencies between proper nouns in the plural and the bad words
def x42(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if ((dependency[0][1]=='NNP$') and (dependency[2][0] in distinct_bad_words)) or ((dependency[2][1]=='NNP$') and (dependency[0][0] in distinct_bad_words)):
            count+=1
    dic = {}
    dic["dependency_proper_noun_plural_bad_words"] = count
    return dic

# count number of dependencies between personal pronouns and the bad words
def x43(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if ((dependency[0][1]=='PRP') and (dependency[2][0] in distinct_bad_words)) or ((dependency[2][1]=='PRP') and (dependency[0][0] in distinct_bad_words)):
            count+=1
    dic = {}
    dic["dependency_personal_pronoun_bad_words"] = count
    return dic

# count number of dependencies between possessive pronouns and the bad words
def x44(list_dependencies):
    # start to count
    count=0
    for dependency in list_dependencies:
        if ((dependency[0][1]=='PRP$') and (dependency[2][0] in distinct_bad_words)) or ((dependency[2][1]=='PRP$') and (dependency[0][0] in distinct_bad_words)):
            count+=1
    dic = {}
    dic["dependency_possessive_pronoun_bad_words"] = count
    return dic

# count number of dependencies between pronouns and the bad words
def x45(list_dependencies):
    count = x41(list_dependencies)["dependency_proper_noun_singular_bad_words"] + x42(list_dependencies)["dependency_proper_noun_plural_bad_words"] + x43(list_dependencies)["dependency_personal_pronoun_bad_words"] + x44(list_dependencies)["dependency_possessive_pronoun_bad_words"]
    dic = {}
    dic["dependency_pronoun_bad_words"] = count
    return dic

def stm_score(sent):
    '''
    Give VADER sentiment score for a comment
    '''
    token_sents = nltk.sent_tokenize(sent)
    res = list(map(sid.polarity_scores,token_sents))
    score = sum([scores['compound'] for scores in res])/len(res)
    return {"sentimental_score":score}

def feature_scaling(features):
    '''
    Rescaling features around means
    features: list of unscaled-feature dictionaries
    '''
    data = []
    sample_mean = {}
    sample_dev = {}
    for d in features:
        data.append(list(d.values()))
    feature_values = np.array(data)
    means = np.mean(feature_values, axis=0)
    stdevs = np.std(feature_values, axis=0)

    for i, k in enumerate(d.keys()):
        sample_mean[k] = means[i]
        sample_dev[k] = stdevs[i]
    for d in features:
        for i,(k,v) in enumerate(d.items()):
            res = (v - means[i])/stdevs[i]
            d[k] = res
    return features

def scale_sample(feature, sample_mean, sample_dev):
    dic = {}
    for k in feature.keys():
        res = (feature[k] - sample_mean[k])/sample_dev[k]
        if (res >= high_threshold):
            dic[k] = 'high'
        elif (res <= low_threshold):
            dic[k] = 'low'
        else:
            dic[k] = 'medium'
    return dic

def bag_of_words(text):
    dic = {}
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if not(word.lower() in stopwords or word in stopwords) and (word.lower() in englishwords or word in englishwords)]
    #tagged = nltk.pos_tag(words, tagset = 'universal')
    fdist = nltk.FreqDist(words)
    for key in fdist.keys():
        dic['contain_{}'.format(key)] = fdist[key]
    return dic
