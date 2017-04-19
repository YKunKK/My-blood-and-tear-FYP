import numpy as np
import pandas as pd


path = 'C:/Users/Administrator/Dropbox/research/FYP/UC_shared_data/'
dat=pd.read_table(path+'docword.kos.txt', sep=' ', skiprows=3, names=['doc', 'word', 'count'])
word_bag = pd.read_table(path+'vocab.kos.txt',sep=' ', header=None)
V=len(word_bag)
D=max(dat['doc'])
K = 3 #total number of topics assigned

# initial matrix of all same entries fail to give meaningful updates, have to use the generated initial value
np.random.seed(np.array(range(K))+1)
beta_initial = np.random.dirichlet(np.ones(V),size=K)
np.random.seed(np.array(range(D))+K+1)
theta_initial = np.random.dirichlet(np.ones(K), size=D)

beta_iteration=[beta_initial]
theta_iteration=[theta_initial]


def w_i_v_d(i, v, d, beta_old, theta_old):
    return theta_old[d-1, i-1]*beta_old[i-1,v-1]/np.sum(theta_old[d-1,]*beta_old.T[v-1,])


def update(i):
    beta_i_temp = []
    for v in range(V):
        document = dat[dat['word'] == (v + 1)]
        # the numerator of all beta_i_*
        beta_i_temp.append(np.sum(np.array(document['count']) * w_i_v_d(i + 1, v + 1, np.array(document['doc']),
                                                                            beta_old=beta_iteration[-1],
                                                                            theta_old=theta_iteration[-1])))
        print v

    theta_i_temp = []
    for d in range(D):
        document = dat[dat['doc'] == (d + 1)]
        # numerators of theta_*_i
        theta_i_temp.append(np.sum(np.array(document['count']) * w_i_v_d(i + 1, np.array(document['word']), d + 1,
                                                                             beta_old=beta_iteration[-1],
                                                                             theta_old=theta_iteration[-1])))
        print d
    return [np.array(beta_i_temp),np.array(theta_i_temp)]



