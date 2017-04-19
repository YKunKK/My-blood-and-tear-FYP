import pandas as pd
import string , re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.tag import StanfordNERTagger
from multiprocessing import Pool, Array
from math import isnan
import numpy as np

path ='/Users/Kun/Desktop/Dropbox/research/twitter_data'

mydiction = ['rt','although','anybody','anywhere','et','etc','q','tu','n','x','c','tl','whenever','xx']



def punc_remover(text):
        token_no_punc = []
        for token in text:
            new_token = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', token)
            if not new_token == ' ':
                token_no_punc.append(new_token)
        # print token_no_punc
        token_no_punc = [word_tokenize(x) for x in
                         token_no_punc]  # do one more tokenization to seperate previous connected words
        return token_no_punc


def tokenization(index):
    if dat.iloc[index]['discount'] >= 0.8:

        try:
            test = dat['text'][index].lower()

            # unicode emotion expression should be removed:
            test = re.sub(r'[^\x00-\x7f]', r' ', test)

            # urls should be removed from the text before tokenization
            # url = []
            # temp_url=[]
            while test.find('https') != -1:
                ind = test.find('https')
                ind2 = test[ind:].find(' ') + ind
                if ind2 != (-1 + ind):
                    # temp_url.append(test[ind:ind2])
                    test = test[:ind] + test[ind2:]
                else:
                    # temp_url.append(test[ind:])
                    test = test[:ind]
            # url.append(temp_url)


            # hashtags should be removed from text  ------------ too hard to extract by hashtag section. keep and tokenization is fine
            test_tag = dat['hashtag'][index].lower()
            tag = []
            if test_tag.strip('[[]]') != '':
                for item in test_tag.strip('[[]]').split(','):
                    item = item.strip()
                    ind = test.find('#' + item)
                    test = test[:ind] + test[ind + len(item) + 1:]
                    tag.append(WordNetLemmatizer().lemmatize(item))

            # use sentance and word tokenization to get seperated words
            test = sent_tokenize(test)
            tokens = [word_tokenize(sent) for sent in test]

            # remove punctuation
            tokens = sum(tokens, [])
            token_no_punc = punc_remover(tokens)

            # remove stop words -----------------  also define my onw dictionary
            token_no_stopwords = []
            for item in token_no_punc:
                if not item == []:
                    for i in range(len(item)):
                        if (not item[i] in stopwords.words('english')) and (not item[i] in mydiction):
                            token_no_stopwords.append(item[i])
            # print token_no_stopwords


            # st = StanfordNERTagger('/Users/Kun/nltk_data/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz',path_to_jar ='/Users/Kun/nltk_data/stanford-ner-2016-10-31/stanford-ner.jar')
            # st.tag(token_no_stopwords)
            # st.tag('rt @unc_basketball: a dance-off at @latenightwroy? pretty cool.\n\na dance-off between @tpinsonn and @joelberryii? yessir! '.split()) # no use for
            # token_tag = [pos_tag(x) for x in token_no_stopwords]
            # print token_tag
            # then use part-of-speech tagging to specify the role of each word in the sentence  #seems not useful up to now
            # the maxent treebank pos tagging model in NLTK is used by default, it is possible to use other model or train own model, but not our focus here


            # test = ' '.join(re.findall('[A-Z][^A-Z]*', test))

            # stemming/lemmatization: return words to its root/general form, e.g. tense, class...; lemmatization also use the context, so can distinguish same word of different meaning, stemming not
            token_clean = map(WordNetLemmatizer().lemmatize, token_no_stopwords)
            # token_clean2 = map(SnowballStemmer('english').stem, token_no_stopwords)

            #further remove unwanted words
            clean=[]
            for i in range(len(token_clean)):
                if (not token_clean[i] in mydiction):
                    clean.append(token_clean[i])

            # return [token_clean, token_clean2]



            #loc = dat.iloc[index]['user_location']
            #try:
                #loc = word_tokenize(re.sub(r'[^\x00-\x7f]', r' ', loc))
                #loc = map(WordNetLemmatizer().lemmatize, loc)
                #loc = sum(punc_remover(loc), [])
            #except:
                #loc = []

            return clean

            # return [token_clean, discount, tag, loc]

        except:
            print index


def indices(lst, targ):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(targ, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def process(year, month, day):

    p=Pool(10)
    result = p.map(tokenization, range(len(dat)))
    p.close()
    p.join()


    word = []
    doc = []
    pointer = 0

    for i in range(len(result)):
        item = result[i]
        try:
            word.extend(item)
            pointer+=1
            doc.extend([pointer]*len(item)) #document index start from 1
        except:
            if item!=None:
                print i


    #combine same word to form vocabulary
    #get vocabulary list, and total frequency in the corpus
    vocabulary = []
    freq = []
    while len(word)>0:
        location = indices(word,word[0])
        vocabulary.append(word[0])
        freq.extend([len(location)])

        [word.pop(i) for i in reversed(location)] #interestingly pop will also delete original word list


    word=[]
    for i in range(len(result)):
        try:
            item = result[i]
            word.extend(item)
        except:
            continue

    #examine and discard too sparse vocabulary:
    freq_remove=[]
    freq_remove_index = sum([indices(freq, 1),indices(freq,2)],[])
    for i in freq_remove_index:
        freq_remove.append(vocabulary[i])
    freq_remove.sort()
    freq_remove_index.sort()
    [vocabulary.pop(i) for i in reversed(freq_remove_index)]
    for item in freq_remove:
        location = indices(word, item)
        [doc.pop(i) for i in reversed(location)]
        [word.pop(i) for i in reversed(location)]


    print len(vocabulary), len(doc), len(word)

    word_index=np.zeros(shape=len(word)).astype('i')
    for i in range(len(vocabulary)):
        word_index[indices(word,vocabulary[i])]= i+1



    pd.DataFrame(data={'doc': doc, 'word':word_index.tolist()}).to_csv(path+'clean_%s_%s_%s' % (year, month, day)+'.txt')
    pd.DataFrame(data = {'vocab':vocabulary}).to_csv(path+'vocab_%s_%s_%s' % (year, month, day)+'.txt')


month = 1
year=2017



for day in range(1,21):
    dat=pd.read_csv(open(path+'/combine_%s_%s_%s' % (str(year), str(month), str(day)) +'.csv','rU'),encoding='utf-8', engine='c')
    print str(month) + '/' + str(day) + 'start'
    process(year, month, day)
    print str(month)+'/'+str(day)+'done'