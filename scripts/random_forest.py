import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.svm import SVC
import numpy as np

INPUT_PATH = "/home/krishna/Dropbox/Frover/inputs/train.csv"
OUTPUT_PATH = "/home/krishna/PycharmProjects/Frover/inputs/breast-cancer-wisconsin.csv"

def read_data(path):
    data = pd.read_csv(path)
    return data

def classifier(features, target, method):
    method.fit(features, target)
    return method

def valid_classifiers(train_X, valid_X, train_y, valid_y):
    results = []
    pred_results = []
    classifiers = [ExtraTreesClassifier(n_estimators=10000, min_weight_fraction_leaf=0.0, min_samples_split=0.2, min_samples_leaf=0.001, max_features='auto', min_impurity_decrease=0.0001, criterion='gini', n_jobs=-1), RandomForestClassifier(1000, n_jobs=-1), LinearDiscriminantAnalysis('lsqr', 0.5)]
    voting_estimators = [('xtrees', clone(classifiers[0])), ('randforr', clone(classifiers[1])), ('lda', clone(classifiers[2]))]
    for method in (classifiers + [AdaBoostClassifier(clone(classifiers[0])), VotingClassifier(estimators=voting_estimators, n_jobs=-1)]):
        method.fit(train_X, train_y)
        results = results + [str(method.score(valid_X, valid_y))]
    print("%s" % (str(results)[1:-1].replace(", ", ",").replace("'","")))

def split_data_set(data_set, train_percentage):
    feature_headers = list(data_set.columns.values)[1:]
    target_header = list(data_set.columns.values)[0]
    train_x, valid_x, train_y, valid_y = train_test_split(data_set[feature_headers], data_set[target_header],
                                                        train_size=train_percentage, test_size=1-train_percentage)
    return train_x, valid_x, train_y, valid_y

def grid_search(clf, params, train_X, valid_X, train_y, valid_y):
	clf = GridSearchCV(estimator = clf, param_grid = params, n_jobs=-1, verbose=True)
	clf.fit(train_X, train_y)

	print("Training score: " , clf.score(train_X, train_y))
	print("Validation Score: ", clf.score(valid_X, valid_y))
	print(clf.best_params_)

	return clf

data = read_data(INPUT_PATH)

train_X, valid_X, train_y, valid_y = split_data_set(data, 0.8)

clf = ExtraTreesClassifier(criterion='gini', n_jobs=-1)


params = [{'n_estimators': [1000, 10000], 'max_features': ['auto', 'sqrt', 'log2', None], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 1,2,3,4,5,6,7,8,9,10], 'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'min_samples_leaf' : [0.1, 0.2, 0.3, 0.4, 0.5]}]

grid_search(clf, params, train_X, valid_X, train_y, valid_y)


