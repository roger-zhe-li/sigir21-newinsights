# -*- coding: utf-8 -*-
import os
import math
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np 
import pandas as pd 
from matplotlib import cm 


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--res_path', nargs='?', default='../results/',
						help='Input data path.')
parser.add_argument('--ds_path', nargs='?', default='../data/',
						help='Input dataset path')
parser.add_argument('--dataset', type=str, default='Epinions',
					choices=['Epinions', 'citeulike', 'ml-10m', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Sports_and_Outdoors'])
parser.add_argument('--loss_type', type=str, default='dcg', 
					choices=['dcg', 'rr', 'ap', 'rbp', 'lambda_dcg', 'lambda_rr', 'lambda_ap', 'lambda_rbp' ],
					help='listwise loss function selection')  
parser.add_argument('--threshold', type=int, default=0,
					help='binary threshold for pos/neg') 
parser.add_argument('--fold', type=int, default=0,
					help='split') 


args = parser.parse_args() 

res_path = args.res_path
# dataset = args.dataset
threshold = args.threshold
# fold = args.fold
ds_path = args.ds_path
# frac = args.frac

# emb_sizes = [8, 16, 32, 64, 128]
emb_size = 32

# loss_types = ['ndcg_5', dcg', 'ap', 'rr', 'rbp_80', 'rbp_90', 'rbp_95']
loss_types = ['dcg', 'ap', 'rr', 'rbp_95']
folds = [0, 1, 2]
# loss_types = ['ap', 'rbp_95']
fracs = [1.0, 2.0, 5.0]
metrics = ['NDCG@5', 'NDCG', 'AP', 'RR', 'RBP_80', 'RBP_90', 'RBP_95']

fig, axes = plt.subplots(3, 7, sharex=True, sharey=True, figsize=(7, 3))
map_vir = cm.get_cmap(name='jet')
colors = [map_vir(i) for i in np.linspace(0, 1, len(loss_types))]

columns = ['user_id', 'freq']

datasets = ['citeulike', 'Epinions', 'Sports_and_Outdoors', 'Home_and_Kitchen']


# for dataset in datasets:
for dataset in datasets:
	if dataset == 'citeulike' or dataset == 'Epinions':
		th = 0
	else:
		th = 4
	ds_file_path = os.path.join(ds_path, dataset, 'th_'+str(th), 'fold_'+str(0), 'train.csv')
	df_ds = pd.read_csv(ds_file_path, header=0)
	user_id = df_ds.user_id.value_counts().sort_index().index.tolist()
	freq = df_ds.user_id.value_counts().sort_index().values.tolist()
	result = {columns[0]: user_id,
			  columns[1]: freq}
	result = pd.DataFrame(result)
	result.to_csv('user_freq_'+dataset+'.csv', index=False)




# fig.savefig('ind_'+dataset+'.png', dpi=300)
		












