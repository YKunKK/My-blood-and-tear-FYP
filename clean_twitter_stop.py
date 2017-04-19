import pandas as pd
import numpy as np

#clean_list=['much', 'also', 'eh', 'whether', 'since', 'someone', 'anyone','sometimes','somebody','even','wo','im','rn','could','go','rn','ever',
#            'yo','instead','ask','something','say','would','anyway','always','might','every','ed','around','know','come','seen','said','went','gone',
#            'take','thing', 'felt','almost','gave','wan','30','50','10','40','0','000','wil','oh','ah','get','ht','100','']

#clean_list= ['13','250','99','80','14',' ']
clean_list=[]
path ='/Users/Kun/Desktop/Dropbox/research/twitter_processed'


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
    dat = pd.read_table(open(path + 'clean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                        skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + 'vocab_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                          skiprows=1, names=['index', 'word'])
    vocab = vocab['word'].tolist()

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


    pd.DataFrame(data={'word': word, 'doc': doc}).to_csv(path + 'clean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt')
    pd.DataFrame(data={'vocab': vocab}).to_csv(path + 'vocab_%s_%s_%s' % (year, month, day) + '.txt')


def deal_with_twitter(year, month, day):
    dat = pd.read_table(open(path + 'clean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                        skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + 'vocab_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'), sep=',',
                          skiprows=1, names=['index', 'word'])
    vocab = vocab['word'].tolist()
    V=len(vocab)

    word=dat['word'].values.tolist()
    doc=dat['doc'].values.tolist()

    #first check if length 1 then dump
    item_to_delete=[]
    for i in range(V):
        if type(vocab[i])==float or len(vocab[i])==1 or len(vocab[i])==0:
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

    [vocab.pop(i) for i in reversed(item_to_delete)]
    vocab=vocab[:]

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

    pd.DataFrame(data={'word': word, 'doc': doc}).to_csv(path + 'clean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt')
    pd.DataFrame(data={'vocab': vocab}).to_csv(path + 'vocab_%s_%s_%s' % (year, month, day) + '.txt')




year=2016
month = 11
for day in range(24, 31):
    deal_with_twitter(year, month, day)

month = 12
for day in range(1, 32):
    try:
        deal_with_twitter(year, month, day)
    except:
        print day

year = 2017
month =1
for day in range(1, 24):
    try:
        deal_with_twitter(year, month, day)
    except:
        print day

#------------------------------------------------------------------