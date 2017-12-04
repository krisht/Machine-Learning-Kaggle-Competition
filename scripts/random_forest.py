import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda, QuadraticDiscriminantAnalysis as qda
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import libsvm_sparse
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import numpy as np

INPUT_PATH = "../inputs/train.csv"
TEST_PATH = "../inputs/test.csv"

def read_data(path, header=1):
	return pd.read_csv(path)

def split_data_set():
	data_set = read_data(INPUT_PATH)
	test_data = read_data(TEST_PATH)
	data_set = shuffle(data_set)
	sets = np.split(data_set, [200], 0)

	valid_y = np.asarray(sets[0]['label'])
	valid_X = np.asarray(sets[0].loc[:, sets[0].columns != 'label'])

	train_y = np.asarray(sets[1]['label'])
	train_X = np.asarray(sets[1].loc[:, sets[1].columns != 'label'])

	test_X = np.asarray(test_data)[:, 1:]

	return train_X, valid_X, test_X, train_y, valid_y


train_X, valid_X, test_X, train_y, valid_y = split_data_set()

clf = RandomForestClassifier(n_jobs=-1)

from scipy.stats import randint as sp_randint
param_dist = {"n_estimators": [10, 100, 1000, 10000], "max_depth": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

n_iter_search = 20
clf = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)



clf.fit(train_X, train_y)
print(clf.best_score_)
print(clf.best_params_)

# print(clf.score(train_X, train_y))
# print(clf.score(valid_X, valid_y))
#
# with open('out.csv', 'w') as f:
# 	f.write('Id,Prediction\n')
#
#
# np.savetxt(open('out.csv', 'ab'), np.c_[np.arange(1, len(test_X)+1), clf.predict(test_X)], '%d', delimiter=',')
#
