import collections
import json
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rbm


def logistic(x):
    """ logistic function """
    return 1.0 / (1 + np.exp(-x))


class RBMRecsys(object):
    
    def __init__(self, N_v, N_h):
        """ init
        Args
            N_v: number of neurons in visible layer
            N_h: number of neurons in hidden layer
        """
        self.N_v = N_v
        self.N_h = N_h
        
        # init weights
        self.W = np.asarray(np.random.RandomState(1234).uniform(
                        low=-0.1 * np.sqrt(6. / (N_h + N_v)),
                        high=0.1 * np.sqrt(6. / (N_h + N_v)),
                        size=(N_v, N_h)))
        self.W = np.insert(self.W, 0, 0, axis=0)
        self.W = np.insert(self.W, 0, 0, axis=1)

    def train_and_recommend(self, user2items_train, user2items_test, top_Ks=[10], epochs=1000, lr=0.1):
        """ train a RBM based recommendation system and do recommendation on test data
        Args
            user2items_train: a matrix of user-items for training
            user2items_test: a matrix of user-items for testing
            top_Ks: a list of top K
            epochs: number of epochs
            lr: learning rate
        Returns
            evaluation metrics
        """
        N_examples = user2items_train.shape[0]
        data = np.insert(user2items_train, 0, 1, axis=1)
        for epoch in range(epochs):
            h_probs = logistic(data.dot(self.W))
            h_probs[:,0] = 1
            h_states = h_probs > np.random.rand(N_examples, self.N_h + 1)
            v_probs_inv = logistic(h_states.dot(self.W.T))
            v_probs_inv[:,0] = 1 
            h_probs_inv = logistic(v_probs_inv.dot(self.W))
            self.W += lr * ((data.T.dot(h_probs) - v_probs_inv.T.dot(h_probs_inv)) / N_examples)
            mean_error = np.sqrt(np.sum((data - v_probs_inv) ** 2) / data.shape[0] / data.shape[1])
            print("Epoch %s: mean error is %s" % (epoch, mean_error))
        return self._evaluate(user2items_train, self.estimate(user2items_train)[:,1:], user2items_test, top_Ks)

    def _evaluate(self, user2items_train, user2items_train_estimated, useridx2itemidx_test, top_Ks):
        """ evaluation on test data
        Args
            user2items_train: a matrix of user-items for training
            user2items_train_estimated: a matrix of user-items for training (estimated)
            useridx2itemidx_test: user index -> item index for testing
            top_Ks: a list of top K
        Returns
            a dict of metric data for each top k
        """
        k2metrics = {}
        for k in top_Ks:
            recalls = []
            precisions = []
            for i, d in enumerate(user2items_train):
                new_pred = set(np.argsort(user2items_train_estimated[i])[-k:]) - set(np.where(d==1)[0])
                hit = float(len(new_pred & useridx2itemidx_test[i]))
                if len(useridx2itemidx_test[i]) > 0:
                    recalls.append(hit / len(useridx2itemidx_test[i]))
                if len(new_pred) > 0:
                    precisions.append(hit / len(new_pred))
            recall = np.mean(recalls)
            precision = np.mean(precisions)
            f1score = 2 * recall * precision / (recall + precision)
            print('\tk=%d recall[%.4f] precision[%.4f] f1score[%.4f]' 
                  % (k, recall, precision, f1score))
            k2metrics[k] = {
                'recall': recall,
                'precision': precision,
                'f1score': f1score
            }
        return k2metrics

    def estimate(self, user2items):
        """ estimate user -> items matrix"""
        N_examples = user2items.shape[0]
        h_states = np.ones((N_examples, self.N_h + 1))
        h_probs = logistic(np.insert(user2items, 0, 1, axis = 1).dot(self.W))
        h_states[:,:] = h_probs > np.random.rand(N_examples, self.N_h + 1)
        h_states = h_states[:,1:]
        return logistic(np.insert(h_states, 0, 1, axis = 1).dot(self.W.T))


def main(data_file, meta_file, output_file):
    """ main entry """
    meta = pd.read_csv(meta_file)

    user2items = pd.read_csv(data_file)
    user2idx = {user_id: i for i, user_id in enumerate(user2items['user_id'].unique())}
    N_users = len(user2idx)
    print('Number of users: %d' % N_users)
    anime2idx = {anime_id: i for i, anime_id in enumerate(user2items['anime_id'].unique())}
    N_animes = len(anime2idx)
    print('Number of animes: %d' % N_animes)

    random.seed(0)
    indices_train = random.sample(range(user2items.shape[0]), int(user2items.shape[0] * 0.7))
    indices_test = list(set(range(user2items.shape[0])) - set(indices_train))
    
    user2items = user2items.values
    data_train = np.zeros((N_users, N_animes))
    for i, idx in enumerate(indices_train):
        if i % 100000 == 0:
            print('%d/%d' % (i, len(indices_train)))
        data_train[user2idx[user2items[idx, 0]], anime2idx[user2items[idx, 1]]] = 1

    useridx2animeidx_test = collections.defaultdict(set)
    for i, idx in enumerate(indices_test):
        useridx2animeidx_test[user2idx[user2items[idx, 0]]].add(anime2idx[user2items[idx, 1]])

    k2metrics = RBMRecsys(N_v=N_animes, N_h=10).train_and_recommend(data_train, useridx2animeidx_test,
        top_Ks=[10,20,30,40,50,60,70,80,90,100], epochs=200, lr=0.1)
    with open(output_file, 'w') as fout:
        json.dump(k2metrics, fout)


if __name__ == '__main__':
    data_file = sys.argv[1]
    meta_file = sys.argv[2]
    output_file = sys.argv[3]
    main(data_file, meta_file, output_file)
