from RBTree import *
import pickle as pkl
import numpy as np

from sklearn.model_selection import RepeatedKFold
import sys 
import time

########
def test_one_time(train_X, train_Y, test_X, test_Y):

	clf = RBTreeClassifier()
	clf.fit(train_X, train_Y)
	Y_predicted = clf.predict(test_X)
	acc = np.mean(Y_predicted == test_Y)

	return acc

def validation(data_name, k_fold, repeat_num=1):
	with open('../RODT/data_pkl/' + data_name + '.pkl', 'rb') as f:
		X, y = pkl.load(f)

	acc_all = np.zeros(k_fold * repeat_num)

	rkf = RepeatedKFold(n_splits=k_fold, n_repeats=repeat_num, random_state=0)
	i = 0
	for train_index, test_index in rkf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		acc_all[i] = test_one_time(X_train, y_train, X_test, y_test)
		i += 1
	
	return acc_all

########
if __name__ == '__main__':
	data_name = 'iris'
	acc_all = validation(data_name, 3, 2)
	print(acc_all)