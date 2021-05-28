# -*- coding: utf-8 -*-
# Import
import os
import math
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import pickle

from guppy import hpy

from eval import evaluation
# from list_loss_relu import dcg_loss, rr_loss, rbp_loss, ap_loss
from Dataset import load_data, train_preparation, test_preparation
from model_lambda import LambdaMF
from utils import *

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

# make possible new folders for data, models, and results
dir_exists('lambda_models')
dir_exists('logs')
dir_exists('lambda_results')

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--data_path', nargs='?', default='./data/',
						help='Input data path.')
parser.add_argument('--save_path', nargs='?', default='./lambda_models/',
                        help='Save data path.')
parser.add_argument('--res_path', nargs='?', default='./lambda_results/',
                        help='Save data path.')
parser.add_argument('--dataset', type=str, default='Epinions',
					choices=['Epinions', 'citeulike', 'ml-10m', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Sports_and_Outdoors'])  
parser.add_argument('--threshold', type=int, default=0,
					help='binary threshold for pos/neg') 
parser.add_argument('--fold', type=int, default=0,
					choices = [0, 1, 2, 3, 4],
					help='fold ID for experiments')
parser.add_argument('--num_pos', type=int, default=20,
					help='number of negative items sampled')
parser.add_argument('--num_neg', type=int, default=200,
					help='number of negative items sampled')
parser.add_argument('--batch_size', type=int, default=16, 
					help='input batch size for training (default: 128)')
parser.add_argument('--random_range', type=float, default=0.01,
					help='[-random_range, random_range] for initialization')
parser.add_argument('--emb_size', type=int, default=32,
					help='latent factor embedding size (default: 32)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--loss_type', type=str, default='lambda_dcg', 
					choices=['lambda_dcg', 'lambda_rr', 'lambda_ap', 'lambda_rbp' ],
					help='listwise loss function selection')
parser.add_argument('--reg', type=float, default=0,
					help='l2 regularization')   
parser.add_argument('--lr', type=float, default=0.1, 
					help='learning rate')  
parser.add_argument('--epochs', type=int, default=120,
					help='number of epochs to train (default: 1000)') 
parser.add_argument('--p', type=float, default=0.95, 
					help='probability value for RBP')
parser.add_argument('--t', type=int, default=2, 
					help='power base for DCG')
parser.add_argument('--b', type=int, default=2, 
					help='log base for DCG')
parser.add_argument('--f_rbp', type=float, default=0.01,
					help='the value to make rankings smaller for RBP training')
parser.add_argument('--temp', type=float, default=1.0,
					help='temperature value for training acceleration')
parser.add_argument('--max_rating', type=float, default=1.0,
					help='max rating scale')
parser.add_argument('--k', type=int, default=5,
					help='cutoff')
parser.add_argument('--frac', type=float, default=1.0,
					help='negative sampling ratio')

args = parser.parse_args()  
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
# print(args.cuda)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


data_path = args.data_path
save_path = args.save_path
res_path = args.res_path
dataset = args.dataset

dir_exists(res_path)
dir_exists(os.path.join(res_path, dataset))


threshold = args.threshold
fold = args.fold
p = args.p
t = args.t
b = args.b 
f_rbp = args.f_rbp 
temp = args.temp
batch_size = args.batch_size

num_pos = args.num_pos
num_neg = args.num_neg 
reg = args.reg
emb_size = args.emb_size
random_range = args.random_range
lr = args.lr
k = args.k

epochs = args.epochs
loss_type = args.loss_type 
max_rating = args.max_rating

frac = args.frac                       


n_user, n_item, train_matrix, test_matrix, test_rel_matrix = load_data(data_path, dataset, threshold, fold)

# print(n_user, n_item)
# print(train_matrix)


# train_neg_items = get_neg_items(n_user, n_item, train_matrix, num_neg)
# print(np.array(train_neg_items).shape)

train_mat, train_rels, train_user = train_preparation(n_user, n_item, train_matrix, frac)
test_mat, test_rels_binary, test_user = test_preparation(n_user, n_item, train_matrix, test_matrix, frac)
_, test_rels_scale, _ = test_preparation(n_user, n_item, train_matrix, test_rel_matrix, frac)
# print(np.array(train_mat).shape)

train_tensor = TensorDataset(torch.from_numpy(np.array(train_user)),
							torch.from_numpy(np.array(train_mat)),
						   torch.from_numpy(np.array(train_rels)))
train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)


test_tensor = TensorDataset(torch.from_numpy(np.array(test_user)),
							torch.from_numpy(np.array(test_mat)),
							torch.from_numpy(np.array(test_rels_binary)),
							torch.from_numpy(np.array(test_rels_scale)))
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

# print(torch.cuda.memory_summary())

# --------------------------------MODEL---------------------------------
model = LambdaMF(n_user, n_item+1, 
			   init_range=random_range, emb_size=emb_size).to(device)

# ---------------------------Train and test---------------------------------------
train_loss_all = []
test_loss_all = []
epoch_all = [] 
# leave an interface for the epochID incase we save the loss value larger than 1
Best_ndcg = Best_ndcg_at_5 = Best_ap = Best_rr = Best_rbp_80 = Best_rbp_90 = Best_rbp_95 = 0
columns = ['loss_type', 'lr', 'threshold', 'reg', 'fold', 'frac', 'emb_size', 'NDCG@5', 'NDCG', 'RR', 'AP', 'RBP_80', 'RBP_90', 'RBP_95']
columns_indi = ['NDCG@5', 'NDCG', 'AP', 'RR', 'RBP_80', 'RBP_90', 'RBP_95']

# --------------------------Define optimizer----------------------------------
best_result = 0
weight_decay = reg

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
 
# model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(p) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(num_neg) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
# save_path = os.path.join(args.save_path, model_file)
# checkpoint = torch.load(save_path)
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# best_result = checkpoint['best_performance']
# epoch = check

for i in tqdm(range(epochs)):

	train_loss = loss_train_test(model, optimizer, train_loader, loss_type, device, \
								t, b, temp, p, f_rbp, num_pos, num_neg)
	# print(train_loss)
	## ------------Evaluation------------------------
	if i == 0 or (i + 1) % 5 == 0:
		NDCG_at_5, NDCG, RR, AP, RBP_80, RBP_90, RBP_95, ndcg_at_5, ndcg, mrr, mAP, rbp_80, rbp_90, rbp_95 = evaluation(model, test_loader, max_rating, device, k, p, n_item)
		if loss_type == 'lambda_dcg':
			result_current = ndcg
		elif loss_type == 'lambda_rr':
			result_current = mrr
		elif loss_type == 'lambda_ap':
			result_current = mAP
		elif loss_type == 'lambda_rbp' and p == 0.80:
			result_current = rbp_80
		elif loss_type == 'lambda_rbp' and p == 0.9:
			result_current = rbp_90
		elif loss_type == 'lambda_rbp' and p == 0.95:
			result_current = rbp_95
# 		else:
# 			print('loss function does not exist.')

		if result_current > best_result:
			epoch = i
			best_result = result_current
			# best_hr = hr
			best_ndcg = ndcg
			best_ndcg_at_5 = ndcg_at_5
			best_mrr = mrr
			best_mAP = mAP
			best_rbp_80 = rbp_80
			best_rbp_90 = rbp_90
			best_rbp_95 = rbp_95
			best_NDCG = NDCG
			best_RR = RR
			best_AP = AP
			best_RBP_80 = RBP_80
			best_RBP_90 = RBP_90
			best_RBP_95 = RBP_95
			model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(p) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(num_neg) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
			save_path = os.path.join(args.save_path, model_file)
			print("Best" + args.loss_type.upper() + ": %.4f" % best_result)
			# save_model(epoch, model, best_result, optimizer, save_path) 
			loss_type = loss_type
			# if loss_type == 'lambda_rbp':
			# 	if p == 0.80:
			# 		loss_type == 'lambda_rbp_80'
			# 	elif p == 0.90:
			# 		loss_typer == 'lambda_rbp_90'
			# 	elif p == 0.95:
			# 		loss_typer == 'lambda_rbp_95'

			result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
			result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
			result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
			if loss_type == 'lambda_rbp':
				name = 'loss_type_' + loss_type + '_' + str(p) + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
			else:
				name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
			res_path = os.path.join('./results/', dataset, 'overall')
			res_path_indi = os.path.join('./results/', dataset, 'individual')

			dir_exists(res_path)
			dir_exists(res_path_indi)
			result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
			result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)


