from os import sep
from numpy.core.fromnumeric import shape
import pandas as pd
#reading user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/recommendation_systems/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
n_users = users.shape[0]

print('Number of users:', n_users)

#reading ratings file
r_cols = ['user_id', 'movid_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('./data/recommendation_systems/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./data/recommendation_systems/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

print('Number of training rates:', rate_train.shape)
print('Number of test rates:', rate_test.shape)

#contruct item profiles

#reading items file
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./data/recommendation_systems/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
n_items = items.shape[0]
print('Number of items:', n_items)

print(items.iloc[0])

X0 = items.values
X_train_counts = X0[:, -19:]

print(X_train_counts[0])


#tfidf

from sklearn.feature_extraction.text import TfidfTransformer

tranformer =  TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = tranformer.fit_transform(X_train_counts.tolist()).toarray()

print(tfidf.shape)

print(tfidf[0])

import numpy as np

def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:, 0] #all user
    # item incices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1; #index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)


#tim mo hinh cho moi user

from sklearn.linear_model import Ridge
from sklearn import linear_model
from math import sqrt
d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):    
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept  = True)
    Xhat = tfidf[ids, :]
    
    clf.fit(Xhat, scores) 
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_    

# predicted scores
Yhat = tfidf.dot(W) + b


n = 99
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])

def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return sqrt(se/cnt)

print ('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print ('RMSE for test    :', evaluate(Yhat, rate_test, W, b))