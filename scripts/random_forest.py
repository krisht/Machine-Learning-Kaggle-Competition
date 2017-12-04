import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda, QuadraticDiscriminantAnalysis as qda
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import libsvm_sparse
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import numpy as np

INPUT_PATH = "../inputs/train.csv"
TEST_PATH = "../inputs/test.csv"

def read_data(path, header=1):
	if header is None:
		data = pd.read_csv(path)
	else:
		data = pd.read_csv(path)
	return data

def split_data_set():
	data_set = read_data(INPUT_PATH)
	test_data = read_data(TEST_PATH)
	data_set = shuffle(data_set)
	sets = np.split(data_set, [200], 0)

	valid_y = np.asarray([sets[0]['label']]).T
	valid_X = np.asarray(sets[0].loc[:, sets[0].columns != 'label'])

	train_y = np.asarray([sets[1]['label']]).T
	train_X = np.asarray(sets[1].loc[:, sets[1].columns != 'label'])

	test_X = np.asarray(test_data)[:, 1:]

	return train_X, valid_X, test_X, train_y, valid_y




train_X, valid_X, test_X,  train_y, valid_y = split_data_set()

clf = lda(solver='eigen', n_components=0, shrinkage=0.10710710710710711)

clf.fit(train_X, train_y)
print(clf.score(train_X, train_y))
print(clf.score(valid_X, valid_y))
with open('out.csv', 'w') as f:
	f.write('Id,Prediction\n')


np.savetxt(open('out.csv', 'ab'), np.c_[np.arange(1, len(test_X)+1), clf.predict(test_X)], '%d', delimiter=',')
#print(clf.predict(test_X))
