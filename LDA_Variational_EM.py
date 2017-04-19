import numpy as np
import pandas as pd
from scipy.special import digamma, loggamma, polygamma
from multiprocessing import Pool
from itertools import chain

path = '/Users/Kun/Desktop/Dropbox/research/FYP/UC_shared_data/'
# path = 'C:/Users/Administrator/Dropbox/research/FYP/UC_shared_data/'
dat = pd.read_table(path + 'docword.kos.txt', sep=' ', skiprows=3, names=['doc', 'word', 'count'])


word_bag = pd.read_table(path + 'vocab.kos.txt', sep=' ', header=None)
V = len(word_bag)  # total number of vocabulary
D = max(dat['doc'])  # total number of documents
K = float(2.0)  # total number of topics assigned

def expand(line):
    return [dat.iloc[line].tolist()[0:2]] * (dat.iloc[line][2])

def expand2(d):
    return range(np.sum(temp.T[0,] == (d + 1)))

# each EM iteration: 1. approximate p(w|alpha, beta) through D iterations; 2. get corresponding alpha(Newton Raphson), beta

# one iteration:
# compute p(w|alpha, beta) through lower bound approximation, for each document d

def L(gamma, phi, alpha, beta, document):  # just for one document d
    w = 0
    temp1 = []
    temp2 = []
    temp3 = []
    digamma_i_one = digamma(gamma)
    digamma_i_sum = digamma(
        sum(gamma))  # derivative of gamma_i of log(gamma(sum_gamma))
    for line in range(len(document)):
        n = document.iloc[line][2]  # count of a particular word
        for line2 in range(n):
            w = w + 1  # index for each word, used for marking phi
            j = document.iloc[line][2]  # index of the particular word
            temp1.append(np.sum(np.log(beta.T[j - 1,]) * phi[w - 1,]))  # sum up zw from 1 to K, for part1

            temp2.append(np.sum(phi[w - 1,] * (digamma_i_one - digamma_i_sum)))

            temp3.append(np.sum(phi[w - 1,] * np.log(phi[w - 1,])))

    part1 = np.sum(temp1)
    part2 = np.sum(temp2)
    part3 = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha)) + np.sum((alpha - 1) * (digamma_i_one - digamma_i_sum))
    part4 = loggamma(np.sum(gamma)) - np.sum(loggamma(gamma)) + np.sum((gamma - 1) * (digamma_i_one - digamma_i_sum))
    part5 = np.sum(temp3)

    return part1 + part2 + part3 - part4 - part5


# iteration to get phi updates, gamma updates, document specific:
def phi_update(gamma, phi, document, beta):
    digamma_i_one = digamma(gamma)
    digamma_i_sum = digamma(np.sum(gamma))

    for w in range(len(document)):
        j = document.iloc[w][1]  # index of the particular word
        temp = np.exp(digamma_i_one - digamma_i_sum) * beta.T[j - 1,]
        phi[w - 1,] = temp / np.sum(temp)

    return phi


def gamma_update(phi, alpha):
    gamma = alpha + np.sum(phi, axis=0)
    return gamma


def lower_bound_parameter(d):
    document = dat_expand[dat_expand['doc'] == d+1]
    Nd = len(document)
    phi = np.ones(shape=(Nd, int(K))) * (1.0 / float(K))
    gamma = Nd / K + alpha

    for iteration in range(max(10,Nd/10)):
        phi = phi_update(gamma, phi, document, beta)
        gamma = gamma_update(phi, alpha)

    return [phi, gamma]


def alpha_newton(alpha_t, gamma):
    h=D * (polygamma(1, np.sum(alpha_t)) - polygamma(1, alpha_t))
    z=D * polygamma(1, np.sum(alpha_t))
    g_at= D * (digamma(np.sum(alpha_t)) - digamma(alpha_t)) + np.sum(digamma(gamma),axis=0) - np.sum(digamma(np.sum(gamma, axis=1)),axis=0)
    c=np.sum(g_at/h)/(1/z+np.sum(1/h))
    U_at=(g_at-c)/h
    return alpha_t-U_at

def eda_newton(eda_t,lambda_):
    f1=K*V*(digamma(V*eda_t)-digamma(eda_t))+np.sum(digamma(lambda_))-V*np.sum(digamma(np.sum(lambda_, axis=1)))
    f2=K*V*V*polygamma(1,V*eda_t)-K*V*polygamma(1,eda_t)
    eda_t_1 = eda_t - f1 / f2
    return eda_t_1

def indices(lst, targ):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(targ, offset + 1)
        except ValueError:
            return result
        result.append(offset)

########################################################################################################################
alpha = np.ones(shape=int(K))*0.1
beta = np.random.dirichlet(size=int(K), alpha=np.ones(V))
eda = 0.1

p = Pool(10)
temp = p.map(expand, range(len(dat)))  # turn count into seperate words
p.close()
p.join()
print 'data expanded -- temp'

temp = np.array(list(chain.from_iterable(temp)))

# construct a column of index for words within each document
p=Pool(10)
w = p.map(expand2,range(D))
p.close()
p.join()
w = np.array(list(chain.from_iterable(w)))

dat_expand = pd.DataFrame(np.vstack((temp.T, np.array(w).T)).T, columns=['doc', 'word', 'w'])
print 'data expanded -- dat + w'
N=len(dat_expand)
del temp
del w

for R in range(5):  # iteration for EM algorithm

    print 'EM iteration:'
    print R

    # get variational parameter for each document, [phi, gamma]
    p=Pool(10)
    r_phi_gamma = p.map(lower_bound_parameter, range(D))
    p.close()
    p.join()

    gamma = []

    for d in range(D):
        gamma.append(r_phi_gamma[d][1])

    phi = np.empty(shape=(int(N),int(K)))
    pointer=0
    for d in range(D):
        l=r_phi_gamma[d][0].shape[0]
        phi[pointer:pointer+l]=r_phi_gamma[d][0]
        pointer+=l

    # then update alpha, beta based on phi, gamma
    for j in range(V):
        document = dat_expand[dat_expand['word'] == (j + 1)]
        temp = []
        for inde in range(len(document)):
            line = document.iloc[inde]
            doc = line['doc']
            w = line['w']  # w is the word index within each document
            phi_d = r_phi_gamma[doc - 1][0]
            temp.append(phi_d[w])
        temp = np.array(temp)

        beta.T[j,] = np.sum(temp, axis=0)  # not normalized

# -------
    lambda_ = beta[:]+eda
# -------

    for i in range(int(K)):
        beta[i,] = beta[i,] / np.sum(beta[i,])

    # update alpha through newton raphson:
    for x in range(10):
        alpha = alpha_newton(alpha_t=alpha, gamma=gamma)

    for y in range(50):
        eda = eda_newton(eda_t=eda,lambda_=lambda_)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————



print alpha
print beta
print eda



import shelve


filename = path+'LDA_Variational_result.out'
my_shelf = shelve.open(filename, 'n')

for key in ['beta','alpha']:#dir():
    try:
        my_shelf[key] = globals()[key]
    except:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
