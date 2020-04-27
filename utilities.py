
# for x20
#cs.cmu
# input: list of word, output: integer
def x20(sent):
    from urllib import request
    url = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words = raw.split('\n')[1:]
    count=0
    for i in range(len(sent)):
        if sent[i] in list_bad_words:
            count+=1
    return count
# input: list of word, output: integer
def x21(sent):
    from urllib import request
    url = 'http://www.bannedwordlist.com/lists/swearWords.txt'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words = raw.split('\r\n')
    count=0
    for i in range(len(sent)):
        if sent[i] in list_bad_words:
            count+=1
    return count
#input: the path to the file, this is a string
def read_local_file(path):   #  for x22
    file1=open(path,'r')
    a= file1.read()
    return a
# input: list of word, output: integer
def x22(sent):
    raw_facebook = read_local_file('facebook-bad-words-list_comma-separated-text-file_2018_07_29.txt')
    facebook_bad_words_list1 = raw_facebook.split(', ')
    facebook_bad_words_list2 = [w for w in facebook_bad_words_list1 if ' ' not in w]
    count=0
    for w in sent:
        if w in facebook_bad_words_list2:
            count+=1
    return count

# input: list of words, output: integer
def x23(sent):
    raw_youtube = read_local_file('facebook-bad-words-list_comma-separated-text-file_2018_07_29.txt')
    youtube_bad_words_list1 = raw_youtube.split(', ')
    youtbe_bad_words_list2 = [w for w in youtube_bad_words_list1 if ' ' not in w]
    count=0
    for w in sent:
        if w in youtbe_bad_words_list2:
            count+=1
    return count

#input: list of words, output:integer
def x24(sent):
    from urllib import request
    url = 'https://gist.githubusercontent.com/ryanlewis/a37739d710ccdb4b406d/raw/0fbd315eb2900bb736609ea894b9bde8217b991a/google_twunter_lol'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    list_bad_words=raw.split('\n')
    count=0
    for w in sent:
        if w in list_bad_words:
            count+=1
    return count

#input: list of words, output:integer
def x25(sent):
    from urllib import request
    url_cmu_20= 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'
    url_swear_21 = 'http://www.bannedwordlist.com/lists/swearWords.txt'
    url_naugty_24 = 'https://gist.githubusercontent.com/ryanlewis/a37739d710ccdb4b406d/raw/0fbd315eb2900bb736609ea894b9bde8217b991a/google_twunter_lol'

    response_20 = request.urlopen(url_cmu_20)
    raw_20 = response_20.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words_20 = raw_20.split('\n')[1:]

    response_21 = request.urlopen(url_swear_21)
    raw = response_21.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words_21 = raw.split('\r\n')

    raw_facebook_22 = read_local_file('facebook-bad-words-list_comma-separated-text-file_2018_07_29.txt')
    facebook_bad_words_list1 = raw_facebook_22.split(', ')
    list_bad_words_22 = [w for w in facebook_bad_words_list1 if ' ' not in w]

    raw_youtube_23 = read_local_file('facebook-bad-words-list_comma-separated-text-file_2018_07_29.txt')
    youtube_bad_words_list1 = raw_youtube_23.split(', ')
    list_bad_words_23 = [w for w in youtube_bad_words_list1 if ' ' not in w]

    response_24 = request.urlopen(url_naugty_24)
    raw_24 = response_24.read().decode('utf8')
    list_bad_words_24 = raw_24.split('\n')

    total_bad_words= list_bad_words_20+list_bad_words_21+list_bad_words_22+list_bad_words_23+list_bad_words_24
    distinct_bad_words=list(set(total_bad_words))

    count=0
    for w in sent:
        if w in distinct_bad_words:
            count+=1
    return count

# input: nothing, output: the list of all bad words from the urls in x20,x21,x22,x23,x24
def list_bad_words():
    from urllib import request
    url_cmu_20 = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'
    url_swear_21 = 'http://www.bannedwordlist.com/lists/swearWords.txt'
    url_naugty_24 = 'https://gist.githubusercontent.com/ryanlewis/a37739d710ccdb4b406d/raw/0fbd315eb2900bb736609ea894b9bde8217b991a/google_twunter_lol'

    response_20 = request.urlopen(url_cmu_20)
    raw_20 = response_20.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words_20 = raw_20.split('\n')[1:]

    response_21 = request.urlopen(url_swear_21)
    raw = response_21.read().decode('utf8')
    # print(raw)  # anal, anus
    list_bad_words_21 = raw.split('\r\n')

    raw_youtube_23 = read_local_file('facebook_bad_words.txt')
    youtube_bad_words_list1 = raw_youtube_23.split(', ')
    list_bad_words_23 = [w for w in youtube_bad_words_list1 if ' ' not in w]

    response_24 = request.urlopen(url_naugty_24)
    raw_24 = response_24.read().decode('utf8')
    list_bad_words_24 = raw_24.split('\n')

    total_bad_words = list_bad_words_20 + list_bad_words_21 + list_bad_words_23 + list_bad_words_24
    distinct_bad_words = list(set(total_bad_words))
    return distinct_bad_words
bad_words=list_bad_words()
print(bad_words)
print(len(bad_words))










