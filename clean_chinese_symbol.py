import pandas as pd
import numpy as np
from ast import literal_eval

#clean_list=[[u'\n'],[u'\u3010'],[u'\u3011'],[u'00'],[u'\u2600'],[u'\u200b'], [u'\u2605'], [u'\u202d'],[u'\u2028'], [u'\u0085'], [u'\u2029']]

#clean_list=[[u'quot']]

#clean_list = [[u'\u4eba'],[u'\ufe0f'],[u'\u4e00\u4e2a'],[u'\u518d'],[u'\u4eca\u5929'],[u'\u6709\u4eba'],[u'\u6700'],[u'\u4e00\u76f4']]

#clean_list=[[u'I'],[u' '],[u'\u6bcf\u5929'],[u'\u540e'],[u'\u51e0\u4e2a'],[u'\u4e00\u573a'],[u'\u6b21'],[u'\u592a'],
#            [u'\u8bf4'],[u'\u53ea'],[u'\u8bb8\u591a'],[u'\u4e2d'],[u'\u4e0b'],[u'\u4e0a'],[u'\u2015'],[u'\u5e74'],[u'\u6708'],[u'\u65e5'],[u'\uff2f']]
clean_list=[[u' ', u' ']]
path ='/Users/Kun/Desktop/Dropbox/research/qq_processed/qq_data'


def indices(lst, targ):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(targ, offset + 1)
        except ValueError:
            return result
        result.append(offset)

def clean_func(year, month, day):
    dat = pd.read_table(open(path + 'clean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                        skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + 'vocab_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                          skiprows=1, names=['index', 'word'])
    vocab = map(literal_eval, vocab['word'])

    word=dat['word'].values.tolist()
    doc=dat['doc'].values.tolist()



    for item in clean_list:
        try:
            position = vocab.index(item)
            V = len(vocab)
            temp = vocab[:position][:]
            temp.append(vocab[-1])
            temp.extend(vocab[(position+1):-1])
            vocab = temp[:]

            del_index = indices(word, position+1)
            del_index.sort()
            [word.pop(i) for i in reversed(del_index)]
            [doc.pop(i) for i in reversed(del_index)]

            for i in indices(word, V):
                word[i] = position+1

        except ValueError:
            continue

    dat = pd.DataFrame(data={'word': word, 'doc': doc})
    D=max(dat['doc'])
    empty = []

    for i in range(1, D + 1):
        if len(dat[dat['doc'] == i]) == 0:
            empty.append(i)
    empty = np.array(empty)

    doc = np.array(dat['doc'][:])
    # then check if empty doc then remove
    for i in range(len(empty)):
        doc[doc >= empty[i]] -= 1
        empty = empty - 1


    pd.DataFrame(data={'word': word, 'doc': doc}).to_csv(path + 'clean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt')
    pd.DataFrame(data={'vocab': vocab}).to_csv(path + 'vocab_qq_%s_%s_%s' % (year, month, day) + '.txt')

year=2016
month = 11
for day in range(23, 31):
    clean_func(year, month, day)

month = 12
for day in range(1, 32):
    try:
        clean_func(year, month, day)
    except:
        print day

year = 2017
month =1
for day in range(1, 24):
    try:
        clean_func(year, month, day)
    except:
        print day














#------------------------------------------------------------------
import pandas as pd
import numpy as np
from ast import literal_eval

path ='/Users/Kun/Desktop/Dropbox/research/qq_processed'

def read_data(year,month,day):
    #read the word_doc matrix, which is already in expand form
    dat = pd.read_table(open(path + '/qq_dataclean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'doc', 'word'])

    N=len(dat)
    D = max(dat['doc'])
    word = dat['word'][:]
    doc = dat['doc'][:]

    return [dat,N,D,word,doc]

def check_doc_index(year, month, day):
    [dat,N,D,word,doc] = read_data(year,month,day)
    #then check if empty doc then remove
    empty = []
    for i in range(1, D + 1):
        if len(dat[dat['doc'] == i]) == 0:
            empty.append(i)
    empty = np.array(empty)

    doc = dat['doc'][:]
    doc = np.array(doc)
    # then check if empty doc then remove
    for i in range(len(empty)):
        doc[doc >= empty[i]] -= 1
        empty = empty - 1

    pd.DataFrame(data={'word': word, 'doc': doc}).to_csv(
        path + 'clean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt')

dat=pd.DataFrame(data={'word': word, 'doc': doc})

D=max(dat['doc'])
for i in range(D):
    if len(dat[dat['doc']==i+1])==0:
        print i


year=2016
month = 11
for day in range(23, 31):
    check_doc_index(year, month, day)

month = 12
for day in range(1, 32):
    try:
        check_doc_index(year, month, day)
    except:
        print day

year = 2017
month =1
for day in range(16, 24):
    try:
        check_doc_index(year, month, day)
    except:
        print day






day=15
# read the word_doc matrix, which is already in expand form
dat = pd.read_table(open(path + '/qq_dataclean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                    skiprows=1, names=['index', 'doc', 'word'])

N = len(dat)
D = max(dat['doc'])
word = dat['word'][:]
doc = dat['doc'][:]

position =0
for d in range(D):
    length = len(dat[dat['doc'] == d + 1])
    if length == 0:
        print d
        doc[position:] -= 1
        D = D - 1
    position = position + length

dat=pd.DataFrame(data={'word': word, 'doc': doc})


for i in range(D):
    length = len(dat[dat['doc'] == i + 1])
    if length == 0:
        print i
pd.DataFrame(data={'word': word, 'doc': doc}).to_csv(
    path + 'clean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt')


#------------------------------------------------------------------
import pandas as pd
import numpy as np
path ='/Users/Kun/Desktop/Dropbox/research/twitter_processed'


def read_data(year,month,day):
    #read the word_doc matrix, which is already in expand form
    dat = pd.read_table(open(path + '/2016/twitter_dataclean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + '/2016/twitter_datavocab_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'word'])

    V= len(vocab)
    N=len(dat)
    D = max(dat['doc'])
    word = dat['word'][:].tolist()
    doc = dat['doc'][:].tolist()

    return [dat,vocab,V,N,D,word,doc]
def indices(lst, targ):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(targ, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def deal_with_twitter(year, month, day):
    [dat,vocab,V,N,D,word,doc] = read_data(year,month,day)

    #first check if length 1 then dump
    item_to_delete=[]
    for i in range(V):
        if type(vocab['word'][i])==float or len(vocab['word'][i])==1 or len(vocab['word'][i])==0:
            item_to_delete.append(i)
            del_index = indices(word, i + 1)
            del_index.sort()
            [word.pop(x) for x in reversed(del_index)]
            word=word[:]
            [doc.pop(x) for x in reversed(del_index)]
            doc=doc[:]
            for j in range(len(word)):
                if word[j]>i+1:
                    word[j]= word[j]-1

    vocab=vocab['word'].tolist()
    [vocab.pop(i) for i in reversed(item_to_delete)]
    vocab=vocab[:]
    pd.DataFrame(data={'vocab': vocab}).to_csv(path + 'vocab_%s_%s_%s' % (year, month, day) + '.txt')

    dat = pd.DataFrame(data={'word': word, 'doc': doc})

    D=max(doc)
    #then check if empty doc then remove
    empty = []
    for i in range(1, D + 1):
        if len(dat[dat['doc'] == i]) == 0:
            empty.append(i)
    empty = np.array(empty)

    doc = dat['doc'][:]
    doc = np.array(doc)
    # then check if empty doc then remove
    for i in range(len(empty)):
        doc[doc >= empty[i]] -= 1
        empty = empty - 1

    pd.DataFrame(data={'doc': doc, 'word': word}).to_csv(
        path + 'clean_%s_%s_%s' % (year, month, day) + '.txt')

month = 11
year=2016
for day in range(24,31):
    deal_with_twitter(year, month, day)



month = 12
year=2016
for day in range(1,32):
    print day
    deal_with_twitter(year, month, day)


def read_data(year,month,day):
    #read the word_doc matrix, which is already in expand form
    dat = pd.read_table(open(path + '/2017/twitter_dataclean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + '/2017/twitter_datavocab_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'word'])

    V= len(vocab)
    N=len(dat)
    D = max(dat['doc'])
    word = dat['word'][:].tolist()
    doc = dat['doc'][:].tolist()

    return [dat,vocab,V,N,D,word,doc]
def deal_with_twitter(year, month, day):
    [dat,vocab,V,N,D,word,doc] = read_data(year,month,day)

    #first check if length 1 then dump
    item_to_delete=[]
    for i in range(V):
        if type(vocab['word'][i])==float or len(vocab['word'][i])==1 or len(vocab['word'][i])==0:
            item_to_delete.append(i)
            del_index = indices(word, i + 1)
            del_index.sort()
            [word.pop(x) for x in reversed(del_index)]
            word=word[:]
            [doc.pop(x) for x in reversed(del_index)]
            doc=doc[:]
            for j in range(len(word)):
                if word[j]>i+1:
                    word[j]= word[j]-1

    vocab=vocab['word'].tolist()
    [vocab.pop(i) for i in reversed(item_to_delete)]
    vocab=vocab[:]
    pd.DataFrame(data={'vocab': vocab}).to_csv(path + 'vocab_%s_%s_%s' % (year+1, month, day) + '.txt')

    dat = pd.DataFrame(data={'word': word, 'doc': doc})

    D=max(doc)
    #then check if empty doc then remove
    empty = []
    for i in range(1, D + 1):
        if len(dat[dat['doc'] == i]) == 0:
            empty.append(i)
    empty = np.array(empty)

    doc = dat['doc'][:]
    doc = np.array(doc)
    # then check if empty doc then remove
    for i in range(len(empty)):
        doc[doc >= empty[i]] -= 1
        empty = empty - 1

    pd.DataFrame(data={'doc': doc, 'word': word}).to_csv(
        path + 'clean_%s_%s_%s' % (year+1, month, day) + '.txt')

month = 1
year=2016
for day in range(1,21):
    print day
    deal_with_twitter(year, month, day)

