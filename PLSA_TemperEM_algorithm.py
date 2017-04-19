import pandas as pd
import numpy as np

from multiprocessing import Pool

# Tempered EM is used as proposed


    path = '/Users/Kun/Desktop/Dropbox/research/FYP/UC_shared_data/'
    #path = 'C:/Users/Administrator/Dropbox/research/FYP/UC_shared_data/'
    dat=pd.read_table(path+'docword.kos.txt', sep=' ', skiprows=3, names=['doc', 'word', 'count'])
    word_bag = pd.read_table(path+'vocab.kos.txt',sep=' ', header=None)
    V=len(word_bag) #total number of vocabulary
    D=max(dat['doc']) #total number of documents
    K = 10 #total number of topics assigned

    #w = np.empty(shape=(K, V, D)) #memory impossible
    def w_i_v_d(i, v, d, mu, beta_old, theta_old):
        return (theta_old[d-1, i-1]*beta_old[i-1,v-1])**mu/np.sum((theta_old[d-1,]*beta_old.T[v-1,])**mu)

    def update(i):
        beta_i_temp = []
        for v in range(V):
            document = dat[dat['word'] == (v + 1)]
            # the numerator of all beta_i_*
            beta_i_temp.append(np.sum(np.array(document['count']) * w_i_v_d(i=i + 1, v=v + 1, d=np.array(document['doc']),
                                                                            mu = mu, beta_old=beta_iteration[-1],
                                                                                theta_old=theta_iteration[-1])))


        theta_i_temp = []
        for d in range(D):
            document = dat[dat['doc'] == (d + 1)]
            # numerators of theta_*_i
            theta_i_temp.append(np.sum(np.array(document['count']) * w_i_v_d(i + 1, np.array(document['word']), d + 1,
                                                                             mu=mu, beta_old=beta_iteration[-1],
                                                                                 theta_old=theta_iteration[-1])))
        print i
        return [np.array(beta_i_temp),np.array(theta_i_temp)]





    # initial matrix of all same entries fail to give meaningful updates, have to use the generated initial value
    np.random.seed(np.array(range(K)) + 100)
    beta_initial = np.random.dirichlet(np.ones(V), size=K)
    np.random.seed(np.array(range(D)) + K + 100)
    theta_initial = np.random.dirichlet(np.ones(K), size=D)

    beta_iteration = [beta_initial]
    theta_iteration = [theta_initial]
    p = Pool(5)

    #when given w_i_v_d(old):
    for mu in [1,1*0.8]:

        for R in range(3):
            beta = np.empty(shape=(K, V))
            theta = np.empty(shape=(D, K))

            # combine the two iterations to speed up the updates
            results = p.map(update, range(K)) # get numerators by i
            for i in range(K):
                beta[i,] = results[i][0]/np.sum(results[i][0])
                theta.T[i,] = results[i][1]

            for d in range(D):
                theta[d,]=theta[d,]/np.sum(theta[d,])

            beta_iteration.append(beta)
            theta_iteration.append(theta)
            print R


#save results
    import shelve


    filename = path+'result.out'
    my_shelf = shelve.open(filename, 'n')

    for key in ['beta_iteration','theta_iteration']:#dir():
        try:
            my_shelf[key] = globals()[key]
        except:
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()




#retrieve results
    filename = path+'result_seed5000_K20_R10.out'
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()


    beta_final = beta_iteration[-1]
    theta_final = theta_iteration[-1]

    topic_pool = []
    for i in range(K):
        index = beta_final[i,].argsort()[-20:][::-1]
        topic_pool.append([word_bag.iloc[index].values.tolist()])

    topic_doc = []
    for d in range(D):
        index = theta_final[d,].argsort()[-5:][::-1]
        topic_doc.append(index) #start with 0










