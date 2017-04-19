import pandas as pd
import bs4
import urllib2
import codecs
import jieba
import string , re
import numpy as np

path ='/Users/Kun/Desktop/Dropbox/research/qq_data'
month = 12
year=2016
day=31
day=20

def indices(lst, targ):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(targ, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def tokenization(index):
    print index

    try:
            test = dat['text'][index]

            # urls should be removed from the text before tokenization
            url = []
            temp_url=[]
            while test.find('http') != -1:
                ind = test.find('http')
                ind2 = test[ind:].find(' ') + ind
                if ind2 != (-1 + ind):
                    temp_url.append(test[ind:ind2])
                    test = test[:ind] + test[ind2:]
                else:
                    temp_url.append(test[ind:])
                    test = test[:ind]
            url.append(temp_url)
            url = sum(url,[])

            if test==' ':
                test=[]
                for url2 in url:
                    try:
                        htmlf = urllib2.urlopen(url2)
                        soup = bs4.BeautifulSoup(htmlf, 'lxml')
                        # res=soup.findAll('div',attrs={'class':'product-unit'})
                        for res in soup.findAll('div', attrs={'class': 'wx2dCode_box_bd clear'}):
                            if res.find('h1', attrs={'class': 'wx2dCode_title'}) != None:
                                test.append(res.find('h1', attrs={'class': 'wx2dCode_title'}).get_text())
                    except:
                        continue
                if len(test)>0:
                    test = test[0]

            if test!=[]:
                #use package jieba to tokenize
                segment = jieba.lcut(test, cut_all=False)

                #remove stopwords and punctuations
                stopwords = codecs.open(path+'/stopword1.txt', 'r', 'utf-8').read().split('\n')
                token_no_stopwords = []
                for seg in segment:
                    if seg not in stopwords and seg !=' ':
                        token_no_stopwords.append(seg)


                token_no_punc = []
                for token in token_no_stopwords:
                    new_token = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', token)
                    if not new_token == ' ':
                        token_no_punc.append(new_token)

                token_no_punc = [jieba.lcut(x, cut_all=False) for x in token_no_punc]


                        # stemming/lemmatization: in Chinese typically do not need lemmatizarion
                #loc = dat.iloc[index]['user_location']
                #try:
                    #loc = word_tokenize(re.sub(r'[^\x00-\x7f]', r' ', loc))
                    #loc = map(WordNetLemmatizer().lemmatize, loc)
                    #loc = sum(punc_remover(loc), [])
                #except:
                    #loc = []

                return token_no_punc

            # return [token_clean, discount, tag, loc]

    except:
        print index

def process(year, month, day):



    #p = Pool(5)
    #result = p.map(tokenization, range(len(dat)))
    #p.close()
    #p.join()

    result = [tokenization(i) for i in range(len(dat))]
    print 'tokenization finish'

    word = []
    doc = []
    pointer = 0

    for i in range(len(result)):
        item = result[i]
        try:
            word.extend(item)
            pointer += 1
            doc.extend([pointer] * len(item))  # document index start from 1
        except:
            if item != None:
                print i

    # combine same word to form vocabulary
    # get vocabulary list, and total frequency in the corpus
    vocabulary = []
    freq = []
    while len(word) > 0:
        location = indices(word, word[0])
        vocabulary.append(word[0])
        freq.extend([len(location)])

        [word.pop(i) for i in reversed(location)]  # interestingly pop will also delete original word list
        if len(word)%1000==0:
            print len(word)

    word = []
    for i in range(len(result)):
        try:
            item = result[i]
            word.extend(item)
        except:
            continue


    # examine and discard too sparse vocabulary:
    freq_remove = []
    freq_remove_index = sum([indices(freq, 1), indices(freq, 2)], [])
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

    word_index = np.zeros(shape=len(word)).astype('i')
    for i in range(len(vocabulary)):
        word_index[indices(word, vocabulary[i])] = i + 1
        if i%3000==0:
            print i

    print 'summarization finish'

    pd.DataFrame(data={'doc': doc, 'word': word_index.tolist()}).to_csv(
        path + 'clean_qq_%s_%s_%s' % (year, month, day) + '.txt')
    pd.DataFrame(data={'vocab': vocabulary}).to_csv(path + 'vocab_qq_%s_%s_%s' % (year, month, day) + '.txt')




#for day in range(15,24):
#    dat = pd.read_csv(open(path + '/combine_qqweibo_%s_%s_%s' % (str(year), str(month), str(day)) + '.csv', 'rU'),
#                      encoding='utf-8', engine='c')
#    tokenization(0)
#    print str(month) + '/' + str(day) + 'start'
#    process(year, month, day)
#    print str(month)+'/'+str(day)+'done'

dat = pd.read_csv(open(path + '/combine_qqweibo_%s_%s_%s' % (str(year), str(month), str(day)) + '.csv', 'rU'),
                  encoding='utf-8', engine='c')
tokenization(0)
print str(month) + '/' + str(day) + 'start'
process(year, month, day)
print str(month) + '/' + str(day) + 'done'