import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from utils import *
# import torchsnooper
import json

def load_data(path, dataset, threshold, fold):
	read_path = os.path.join(path, dataset, 'th_'+str(threshold))
	data_path = os.path.join(read_path, 'fold_'+str(fold), 'train.csv')
	stat_path = os.path.join(read_path, 'dataset_meta_info_'+str(threshold)+'.json')
	test_path = os.path.join(read_path, 'fold_'+str(fold), 'test.csv')

	df = pd.read_csv(data_path, header=0)
	df_test = pd.read_csv(test_path, header=0)
	print(df.shape)
	print(df_test.shape)

	# print(df.shape, df_test.shape)

	with open(os.path.join(stat_path), 'r') as f:
		dataset_meta_info = json.load(f)


	n_user = dataset_meta_info['user_size']
	n_item = dataset_meta_info['item_size']
	
	train_row = []
	train_col = []
	train_rating = []

	for line in df.itertuples():
		# print(line)
		if line[3] >= threshold:
			u = line[4]
			i = line[5]
			train_row.append(u)
			train_col.append(i)
			train_rating.append(1)
	train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_user, n_item))

	test_row = []
	test_col = []
	test_rating = []

	for line in df_test.itertuples():
		if line[3] >= threshold:
			u = line[4]
			i = line[5]
			test_row.append(u)
			test_col.append(i)
			test_rating.append(1)
	test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_user, n_item))

	test_row = []
	test_col = []
	test_rel = []

	for line in df_test.itertuples():
		u = line[4]
		i = line[5]
		rel = line[3]
		test_row.append(u)
		test_col.append(i)
		test_rel.append(rel)
	test_rel_matrix = csr_matrix((test_rel, (test_row, test_col)), shape=(n_user, n_item))

	return n_user, n_item, train_matrix, test_matrix, test_rel_matrix


def train_preparation(n_user, n_item, matrix, frac):
	# num_neg: number of items per user
	# user_row: userIDs in the training set

	mat = []
	rels = []
	users = []

	all_items = set(np.arange(n_item))
	for user in range(n_user):
		pos_items = list(matrix.getrow(user).nonzero()[1])
		# neg_pool = neg_items[user]
		neg_pool = list(all_items - set(matrix.getrow(user).nonzero()[1]))

		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)

		# if len_pos > m:
		# 	m = len_pos

		# pos_p2 = []

		train_rel = [1] * len_pos  + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))
		items = pos_items + neg_i


		mat.append(items)
		rels.append(train_rel)
		users.append(user)


	max_cols = max([len(item) for item in mat])
	for line in mat:
		line += [n_item] * (max_cols - len(line))
	for line in rels:
		line += [20] * (max_cols - len(line))


	# print(np.array(mat).dtype)
	return mat, rels, users


def test_preparation(n_user, n_item, train_matrix, test_matrix, frac):
	# num_neg: number of items per user
	# user_row: userIDs in the training set

	mat = []
	rels = []
	users = []

	all_items = set(np.arange(n_item))

	for user in range(n_user):
		pos_items = list(test_matrix.getrow(user).nonzero()[1])
		# neg_pool = neg_items[user]
		neg_pool = list(all_items - set(train_matrix.getrow(user).nonzero()[1]) - set(test_matrix.getrow(user).nonzero()[1]))

		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)

		test_rel = [1] * len_pos + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))

		items = pos_items + neg_i
		# test_rel = [1] * len_pos + [0] * len(neg_pool)
		
		# items = pos_items + neg_pool

		mat.append(items)
		rels.append(test_rel)
		users.append(user)

	max_cols = max([len(item) for item in mat])
	for line in mat:
		line += [n_item] * (max_cols - len(line))
	for line in rels:
		line += [20] * (max_cols - len(line))
	# print(np.array(mat).shape)
	# print(np.array(rels).shape)


	return mat, rels, users





