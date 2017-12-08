import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda, QuadraticDiscriminantAnalysis as qda
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import libsvm_sparse
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.utils import shuffle
from scipy.stats import randint as sp_randint
import numpy as np
from xgboost import XGBClassifier

from sklearn.mixture import GaussianMixture as GMM

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


# Model
clf = LogisticRegression()

# Search Parameters

param_dist = {'penalty':['l1', 'l2'], 'solver':['liblinear', 'sag', 'saga'], 'n_jobs':[8], 'C':[0.5, 1.0, 1.5]}

#param_dist = {'objective': ['multi:softmax'], 'n_estimators':[1000], 'max_depth':[5]} #number of trees, change it to 1000 for better results}

grid_search = True

if grid_search:
	clf = GridSearchCV(clf, param_grid=param_dist)
	clf.fit(train_X, train_y)
	print(clf.best_score_)
	print(clf.best_params_)
else:
	clf.fit(train_X, train_y)
	print("Training Score: ", clf.score(train_X, train_y))
	print("Validation Score: ", clf.score(valid_X, valid_y))
	with open('out.csv', 'w') as f:
		f.write('Id,Prediction\n')
	np.savetxt(open('out.csv', 'ab'), np.c_[np.arange(1, len(test_X)+1), clf.predict(test_X)], '%d', delimiter=',')