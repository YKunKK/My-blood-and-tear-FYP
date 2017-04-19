import numpy as np
import pandas as pd
from random import randint,seed
from scipy.stats import hmean
from scipy.special import gammaln
from multiprocessing import Pool
from itertools import chain
import timeit
import matplotlib.pyplot as plt

path = '/Users/Kun/Desktop/Dropbox/research/FYP/UC_shared_data/'
dat = pd.read_table(path + 'docword.kos.txt', sep=' ', skiprows=3, names=['doc', 'word', 'count'])
#the dat already have the format we want

word_bag = pd.read_table(path + 'vocab.kos.txt', sep=' ', header=None)
V = len(word_bag)  # total number of vocabulary
D = max(dat['doc'])  # total number of documents
K = float(20)  # total number of topics assigned
N = int(np.sum(dat['count']))
likelihood = []

def expand(line):
    return [dat.iloc[line].tolist()[0:2]] * (dat.iloc[line][2])

p= Pool(15)
temp = p.map(expand, range(len(dat)))  # turn count into seperate words
p.close()
p.join()
print 'data expanded -- temp'

temp = np.array(list(chain.from_iterable(temp)))

dat= pd.DataFrame(temp, columns=['doc', 'word'])
print 'data expanded -- each word is seperately viewed'

del temp

word = dat['word'].values
doc = dat['doc'].values


#parameter alpha and beta
alpha = 0.1
beta = 0.1

#initialize vector z
def func1(x):
    seed(x)
    return randint(1,K)

p= Pool(15)
Z = np.array(p.map(func1,range(N)))
p.close()
p.join()


# to reduce computation, we can save all the nj_wi and only update after we have updated Z:
nj = np.zeros(shape=(int(K), V))
mj = np.zeros(shape=(int(K), int(D)))

def funcnj(w):
    return len(topic_j[topic_j['word'] == (w + 1)])
def funcmj(d):
    return len(topic_j[topic_j['doc'] == (d + 1)])


for j in range(1, int(K) + 1):
    topic_j = dat[Z == j]  # from topic j
    p1= Pool(15)
    nj[j - 1] = p1.map(funcnj,range(int(V)))
    p1.close()
    p1.join()

    p2 = Pool(15)
    mj[j - 1] = p2.map(funcmj,range(int(D)))
    p2.close()
    p2.join()
    print j


nj = nj.astype('i')
mj = mj.astype('i')
nj_s = np.sum(nj,axis=1).astype('i') #sum by each topic j
mj_s = np.sum(mj,axis=0).astype('i') #sum by each document d


if np.sum(mj)!=N:
    print 'Error'



#####attention: if do Gibbs update one by one, too time consuming: ~5min per iteration, not feasible
#####with reference to other sources, the common practise is do one update for whole sample

# update whole sample:
# to get the whole update from previous Z:
def Z_update(i):

    #Z_old = int(Z_share[:][i])
    Z_old = int(Z[i])
    wi = int(word[i])
    di = int(doc[i])

    #nj_wi = np.array(nj_shared[:][int((wi-1)*K):int(wi*K)])  # under word wi
    nj_wi = nj[:,wi-1].tolist()
    nj_wi[Z_old - 1] -= 1
    #nj_sum = np.array(nj_s_shared[:])
    nj_sum = nj_s.tolist()
    nj_sum[Z_old - 1] -= 1


    #mj_di = np.array(mj_shared[:][int((di-1)*K):int(di*K)]) # under doc di
    mj_di = mj[:, di-1].tolist()
    mj_di[Z_old - 1] -= 1
    #msum_di = mj_s_shared[:][int(di) - 1] - 1
    #msum_di = mj_s.tolist()[di-1] -1

    p = ((np.array(nj_wi) + beta) / (np.array(nj_sum) + V * float(beta))) * ((np.array(mj_di) + alpha)) #/ (msum_di + K * float(alpha)))

    p = p / np.sum(p)

    try:
        return [i, np.random.choice(range(1, (int(K) + 1)), p = p)]
    except:
        print 'error'+str(i)+str(p)

def count_update_shared(i):
    Z_old = Z[i]
    Z_n = Z_result[i]
    wi = int(word[i])
    di = int(doc[i])
    nj[Z_old-1][wi-1] -=1
    nj[Z_n - 1][wi - 1] +=1
    mj[Z_old-1][di-1] -=1
    mj[Z_n - 1][di - 1] += 1

def log_like_w_z():
    return K*(gammaln(float(beta)*V)-V*gammaln(beta))+ np.sum(gammaln(float(beta)+nj)) - np.sum(gammaln(nj_s+float(V)*beta))

likelihood_w_z =[]

its = 300 #total number of iterations to do Gibbs sampling, seems ~200 the chain stablize, and another 100 for doing sampling

for iteration in range(its):
    print 'iteration = '+str(iteration)


    start = timeit.default_timer()

    p = Pool(10)
    Z_result = p.map(Z_update,range(int(N)))#one update needs 2min
    p.close()
    p.join()

    temp = np.array(Z_result).T
    temp2 = np.argsort(temp[0])
    Z_result = temp[:,temp2][1]

    [count_update_shared(i) for i in range(int(N))] #interestingly I can not use a pool type without error
    nj_s = np.sum(nj,axis=1).astype('i')

    Z = Z_result



    #also compute the log-likelihood p(w|z)*p(z) (only the changing part)

    l= np.sum(gammaln(nj+beta))- np.sum(gammaln(nj_s+V*beta))+ np.sum(gammaln(alpha+mj))-np.sum(gammaln(float(K)*alpha+mj_s))

    print 'likehood: ' + str(l)

    likelihood.append(l)
    likelihood_w_z.append(log_like_w_z())

    stop = timeit.default_timer()

    print 'time'+str(stop - start)



#store the sampled data:
path = '/Users/Kun/Desktop/Dropbox/research/FYP/UC_shared_data/K_20/'


pd.DataFrame(data={'likelihood_full': likelihood}).to_csv(path + 'likelihood_full' + '.txt')
pd.DataFrame(data={'likelihood_w_given_z':likelihood_w_z}).to_csv(path + 'likelihood_w_given_z' + '.txt')
pd.DataFrame(data=mj).to_csv(path + 'mj' + '.txt')
pd.DataFrame(data=nj).to_csv(path + 'nj' + '.txt')
pd.DataFrame(data=Z).to_csv(path + 'Z' + '.txt')


plt.plot(likelihood)
plt.plot(likelihood_w_z)

#for number of topic selection, need use harmonic mean to estimate observed likelihood:

    #need to sample from the chain
sample = likelihood_w_z[(len(likelihood_w_z)-95)::10]
estimate_w = np.mean(sample) + np.log(hmean(np.exp(np.mean(sample)-sample)))
text_file = open(path+"P(w).txt", "w")
text_file.write(str(estimate_w))
text_file.close()

#for final estimation of topics and distribution:
phi = np.empty(shape=(int(K), int(V)))
theta = np.empty(shape=(int(K), int(D)))
for i in range(int(K)):
    phi[i] = (nj[i]+beta)/(nj_s[i]+float(V)*beta)
    theta[i]=(mj[i]+alpha)/(mj_s+float(K*alpha))