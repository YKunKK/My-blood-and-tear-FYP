import pandas as pd
import numpy as np
import ast
from random import randint,seed
from scipy.stats import hmean
from scipy.special import gammaln
from multiprocessing import Pool
from itertools import chain
import timeit
import matplotlib.pyplot as plt

K=50.0
c=0.5
alpha_base=20/float(K)
beta_base=0.1
path ='/Users/Kun/Desktop/Dropbox/research/twitter_processed'
alpha, beta, phi_old, theta_old, Z = None, None, None, None, None
threshold = 0.3


year=2016
month = 11
day=24

#consider if only process the current and incoming corpus:

def read_data(year,month,day):
    #read the word_doc matrix, which is already in expand form
    dat = pd.read_table(open(path + '/qq_dataclean_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + '/qq_datavocab_qq_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'word'])
    vocab = map(ast.literal_eval,vocab['word'])

    V= len(vocab)
    N=len(dat)
    D = max(dat['doc'])
    word = dat['word'][:]
    doc = dat['doc'][:]

    return [dat,vocab,V,N,D,word,doc]

def read_data_T(year,month,day):
    #read the word_doc matrix, which is already in expand form
    dat = pd.read_table(open(path + 'clean_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'doc', 'word'])
    vocab = pd.read_table(open(path + 'vocab_%s_%s_%s' % (str(year), str(month), str(day)) + '.txt'),  sep=',', skiprows=1, names=['index', 'word'])
    vocab = vocab['word'].tolist()
    V= len(vocab)
    N=len(dat)
    D = max(dat['doc'])
    word = dat['word'][:]
    doc = dat['doc'][:]

    return [dat,vocab,V,N,D,word,doc]


def func1(x):
    seed(x)
    return randint(1,K)

def funcnj(w):
    return len(topic_j[topic_j['word'] == (w + 1)])
def funcmj(d):
    return len(topic_j[topic_j['doc'] == (d + 1)])

def njmj(i):
    loc = dat.iloc[i]
    j=Z[i]
    nj[j-1][loc[2]-1] +=1
    mj[j-1][loc[0]-1] +=1

def Z_update(i):

    Z_old = int(Z[i])
    wi = int(word[i])
    di = int(doc[i])

    nj_wi = nj[:,wi-1][:]
    nj_wi[Z_old - 1] -= 1
    nj_sum = nj_s[:]
    nj_sum[Z_old - 1] -= 1


    mj_di = mj[:, di-1][:]
    mj_di[Z_old - 1] -= 1

    p = ((nj_wi + beta.T[wi-1]) / (nj_sum + beta_s) * (mj_di + alpha.T[di-1]))
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

def log_like_w_z():     #log likelihood of W give Z
    return np.sum(gammaln(beta_s)-np.sum(gammaln(beta)))+ np.sum(gammaln(beta+nj)) - np.sum(gammaln(nj_s+beta_s))

def phi_theta_temp():
    for i in range(int(K)):
        phi_temp[i] = (nj[i] + beta[i]) / (nj_s[i] + beta_s[i])
        theta_temp[i] = (mj[i] + alpha[i]) / (mj_s + np.sum(alpha[i]))


###################################################################################################################
#the following start the iteration

[dat1,vocab1,V1,N1,D1,word1,doc1] = read_data_T(year,month,day) #######################################################
[dat2,vocab2,V2,N2,D2,word2,doc2] = read_data_T(year,month,day+1)   #######################################################

#modify the dynamic vocabulary:
duplicate_2in1= []
duplicate_2in2 = []
vocab_full = vocab1[:]                  #to combine the two vocabulary list
vocab2_modify_index = np.array(range(V2))+1 #the new position index when considering vocab1 words
for i in range(V2):
    try:
        temp = vocab1.index(vocab2[i])
        duplicate_2in1.append(temp)      #store the position the duplicate in 1
        duplicate_2in2.append(i)         #store the position in 2
        vocab2_modify_index[i]=temp+1
    except ValueError:
        vocab_full.append(vocab2[i])
        vocab2_modify_index[i]=len(vocab_full)
        continue


word2_new=[vocab2_modify_index[word2.values[i]-1] for i in range(len(word2))]        #update word index in doc2

#then combine the two corpus
dat = pd.DataFrame({'index':dat1['index'].append(dat2['index']),'doc':doc1.append(dat2['doc']+D1),'word':word1.append(pd.Series(word2_new))})
word = dat['word'].values.astype('i')
doc=dat['doc'].values.astype('i')
V=len(vocab_full)
N=len(dat)
D=D1+D2

if alpha==None and beta==None:
    alpha = np.ones(shape=(int(K),int(D)))*alpha_base
    beta = np.ones(shape=(int(K), int(V)))*beta_base

beta_s = np.sum(beta, axis=1).astype('d')
alpha_s = np.sum(alpha, axis=0).astype('d')

#initialize, Z, nj, mj, if Z is not obtained from previous stage:
if Z==None:
    p= Pool(15)
    Z = np.array(p.map(func1,range(N)))         #initialize vector z
    p.close()
    p.join()

nj = np.zeros(shape=(int(K), V))
mj = np.zeros(shape=(int(K), int(D)))

[njmj(i) for i in range(N)]

#for j in range(1, int(K) + 1):
#    topic_j = dat[Z == j]  # from topic j
#    p1= Pool(30)
#    nj[j - 1] = p1.map(funcnj,range(int(V)))
#    p1.close()
#    p1.join()
#
#    p2 = Pool(30)
#    mj[j - 1] = p2.map(funcmj,range(int(D)))
#    p2.close()
#    p2.join()
#    print j

nj = nj.astype('i')
mj = mj.astype('i')
nj_s = np.sum(nj,axis=1).astype('i') #sum by each topic j
mj_s = np.sum(mj,axis=0).astype('i') #sum by each document d


likelihood_w_z =[]
likelihood = []
phi = np.zeros(shape=(int(K), int(V)))
theta = np.zeros(shape=(int(K), int(D)))
phi_temp = np.empty(shape=(int(K), int(V)))
theta_temp = np.empty(shape=(int(K), int(D)))
its = 350   #total number of iterations to do Gibbs sampling, seems ~300 the chain stablize, and another 100 for doing sampling

for iteration in range(its):

    p = Pool(20)
    Z_result = p.map(Z_update,range(int(N)))#one update needs 2min
    p.close()
    p.join()

    temp = np.array(Z_result).T
    temp2 = np.argsort(temp[0])
    Z_result = temp[:,temp2][1]

    [count_update_shared(i) for i in range(int(N))]
    nj_s = np.sum(nj,axis=1).astype('i')

    Z = Z_result
    l= np.sum(gammaln(beta_s)-np.sum(gammaln(beta))) + np.sum(gammaln(alpha_s)-np.sum(gammaln(alpha))) + np.sum(gammaln(nj+beta))- np.sum(gammaln(nj_s+beta_s))+ np.sum(gammaln(alpha+mj))-np.sum(gammaln(alpha_s+mj_s))

    likelihood.append(l)
    likelihood_w_z.append(log_like_w_z())

    print 'iteration = ' + str(iteration) +', likehood: ' + str(l)

    if iteration%2==0:
        phi_theta_temp()
        phi=phi+phi_temp
        theta=theta+theta_temp


    #using 350 iterations, and use the last 26 samples (every other sample) to average for phi and theta estimation
    if iteration>=299 and iteration%2==1:
        phi_theta_temp()
        phi=phi+phi_temp
        theta=theta+theta_temp


#we can retain by division, but if only look at the ranking for picking out words and topics, the division is not necessary
dup=(its-1-299)/2+1
phi=phi/float(dup)
theta=theta/float(dup)

#store the sampled data:
path_store = path+'_K%s_%s_%s_%s' % (str(int(K)), str(year), str(month), str(day))

pd.DataFrame(data={'likelihood_full': likelihood}).to_csv(path_store + 'likelihood_full' + '.txt')
pd.DataFrame(data={'likelihood_w_given_z':likelihood_w_z}).to_csv(path_store + 'likelihood_w_given_z' + '.txt')
pd.DataFrame(data=mj.T).to_csv(path_store + 'mj' + '.txt')
pd.DataFrame(data=nj.T).to_csv(path_store + 'nj' + '.txt')
pd.DataFrame(data=Z).to_csv(path_store + 'Z' + '.txt')

del likelihood,likelihood_w_z
#to retrieve the stored data:
likelihood = pd.read_table(open(path_store + 'likelihood_full' + '.txt'), sep=',')['likelihood_full'].tolist()
likelihood_w_z = pd.read_table(open(path_store + 'likelihood_w_given_z' + '.txt'), sep=',')['likelihood_w_given_z'].tolist()
mj = pd.read_table(open(path_store + 'mj' + '.txt'), sep=',').as_matrix().T[1:].astype('i')
nj= pd.read_table(open(path_store + 'nj' + '.txt'), sep=',').as_matrix().T[1:].astype('i')
Z= np.array(pd.read_table(open(path_store + 'Z' + '.txt'),sep=',')['0'])
nj_s = np.sum(nj,axis=1).astype('i') #sum by each topic j
mj_s = np.sum(mj,axis=0).astype('i') #sum by each document d



#to look at which topic mostly mentioned, we count the topics with highest theta values, and up to the sum to 0.8
doc_topic_importance=np.zeros(shape=(int(D), int(K)))
for d in range(D):
    position = theta.T[d].argsort()[-1:-3:-1]
    doc_topic_importance[d][position]+=1

topic_importance=np.sum(doc_topic_importance, axis=0)

print "most importance 10 topics " +str(topic_importance.argsort()[-1:-11:-1] +1 )
print "least important 10 topics " +str(topic_importance.argsort()[0:10:1] +1 )


#visualize the topics, look at the top 40 words of the topics
for i in topic_importance.argsort()[-1:-6:-1]:
    print i+1
    position = phi[i].argsort()[-1:-41:-1]
    print np.sum(phi[i][phi[i].argsort()[-1:-41:-1]])
    for j in position.tolist():
        print vocab_full[j]#[0]


for i in topic_importance.argsort()[0:5:1]:
    print i+1
    position = phi[i].argsort()[-1:-41:-1]
    print np.sum(phi[i][phi[i].argsort()[-1:-41:-1]])
    for j in position.tolist():
        print vocab_full[j]#[0]

del doc_topic_importance, topic_importance

for i in range(int(K)):
    position = phi[i].argsort()[-1:-41:-1]
    if np.sum(phi[i][phi[i].argsort()[-1:-41:-1]]) >=0.55:
        print i

#################################################################################################################

#to identify the novel topics:
JSD_new = np.empty(shape=(int(K), int(K)))


if phi_old != None:
    #here need to realian the vocabulary word sequence in order to do comparison:
    #in fact only the vocab1 = old_vocab2 as the overlap
    #so the new phi only need to keep up to V1

    #for the old vocab, retrieve the position index
    position_old=(np.array(position_old)-1).tolist()

    JSD_inter = np.empty(shape=(int(K), int(K)))
    for i in range(int(K)):
        for j in range(int(K)):
            P1=phi_old[i][position_old]
            Q1=phi[i][:V1]
            P2=phi_old[j][position_old]
            Q2=phi[j][:V1]

            #JSD_new[i][j]=0.5*kld(Q1,0.5*(Q1+Q2))+0.5*kld(Q2,0.5*(Q1+Q2))
            JSD_inter[i][j]=0.5*kld(P1,0.5*(P1+Q2))+0.5*kld(Q2,0.5*(P1+Q2))
else:
    for i in range(int(K)):
        for j in range(int(K)):
            Q1=phi[i]
            Q2=phi[j]
            JSD_new[i][j]=0.5*kld(Q1,0.5*(Q1+Q2))+0.5*kld(Q2,0.5*(Q1+Q2))



#print JSD_new

print JSD_inter
#print JSD_old


threshold=0.44
new_topic_detect=np.zeros((int(K), int(K)))

for i in range(int(K)):
    for j in range(int(K)):
        if JSD_new[i][j]>threshold:
            new_topic_detect[i][j]=1

np.sum(new_topic_detect,axis=1)


#JSD_old = JSD_new.copy()
del JSD_new
del new_topic_detect
if phi_old != None:
    del JSD_inter


#################################################################################################################
phi_old = phi.copy()
theta_old = theta.copy()

del phi, theta
#then need to update alpha and beta based on historical information
#alpha:
D_old = D        #the 'old' processed document refers to all in the last window (2 corpus)
N_old = N

#for seen documents : d= d' to end
alpha_j_d = (mj.T[D1:]*alpha_base*D_old*K/float(N_old)).T * c + (1-c)*alpha_base*np.ones(shape=(int(K),int(D2)))


#need to read in the new dataset
#[dat3,vocab3,V3,N3,D3,word3,doc3] = read_data_T(year,month,day+2)
[dat3,vocab3,V3,N3,D3,word3,doc3] = read_data_q(year,month,day+2)

alpha_new = alpha_base*np.ones(shape=(int(K),int(D3)))
alpha = np.array(np.concatenate((alpha_j_d, alpha_new), axis=1))

del alpha_new, alpha_j_d, dat3

#beta:
#modify the dynamic vocabulary of doc 2 and doc 3:
vocab_full_23= vocab2[:]                  #to combine the two vocabulary list
vocab3_modify_index = np.array(range(V3))+1 #the new position index when considering vocab1 words
for i in range(V3):
    try:
        temp = vocab2.index(vocab3[i])
        vocab3_modify_index[i]=temp+1
    except ValueError:
        vocab_full_23.append(vocab3[i])
        vocab3_modify_index[i]=len(vocab_full_23)
        continue

V_new = len(vocab_full_23)
beta= np.ones(shape=(int(K), int(V_new)))*beta_base


#beta is for topic_word distribution prior, based on vocab word
#only the first half are the seen vocabulary
for i in range(V2):
    w_index_vocab12 = vocab_full.index(vocab_full_23[i])
    w_index_vocab23 = i
    beta.T[w_index_vocab23] = beta_base*(1-c)+beta_base*c*nj.T[w_index_vocab12]*K*V_new/float(N_old)


#we can further keep the topic assignemnts for the old documents as the initial stage
p= Pool(15)
Z_D3 = np.array(p.map(func1,range(N3)))         #initialize vector z
p.close()
p.join()

Z = Z[N1:].tolist().extend(Z_D3)

day=day+1

for i in range(V2):
    vocab_full.index(vocab2[i])
vocab_full_old = vocab_full[:]

position_old=vocab2_modify_index.tolist()[:]
del nj,mj,nj_s,mj_s,
del vocab3,V3,N3,D3,word3,doc3,Z_D3,vocab_full_23,vocab3_modify_index, vocab_full, vocab2_modify_index, w_index_vocab12, w_index_vocab23
#################################################################################################################



