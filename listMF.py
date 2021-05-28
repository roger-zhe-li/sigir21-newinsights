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
from model import ListMF
from utils import *

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

# make possible new folders for data, models, and results
dir_exists('models')
dir_exists('logs')
dir_exists('results')

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--data_path', nargs='?', default='./data/',
						help='Input data path.')
parser.add_argument('--save_path', nargs='?', default='./models/',
                        help='Save data path.')
parser.add_argument('--res_path', nargs='?', default='./results/',
                        help='Save data path.')
parser.add_argument('--dataset', type=str, default='Epinions',
					choices=['Epinions', 'citeulike', 'ml-10m', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Sports_and_Outdoors', 'Instant_Video'])  
parser.add_argument('--threshold', type=int, default=0,
					help='binary threshold for pos/neg') 
parser.add_argument('--fold', type=int, default=0,
					choices = [0, 1, 2, 3, 4],
					help='fold ID for experiments')
parser.add_argument('--num_pos', type=int, default=20,
					help='number of negative items sampled')
parser.add_argument('--num_neg', type=int, default=200,
					help='number of negative items sampled')
parser.add_argument('--batch_size', type=int, default=2048, 
					help='input batch size for training (default: 128)')
parser.add_argument('--random_range', type=float, default=0.01,
					help='[-random_range, random_range] for initialization')
parser.add_argument('--emb_size', type=int, default=5,
					help='latent factor embedding size (default: 32)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--loss_type', type=str, default='dcg', 
					choices=['dcg', 'rr', 'ap', 'rbp', 'nrbp', 'nrbp_1' ],
					help='listwise loss function selection')
parser.add_argument('--reg', type=float, default=0,
					help='l2 regularization')   
parser.add_argument('--lr', type=float, default=0.1, 
					help='learning rate')  
parser.add_argument('--epochs', type=int, default=3000,
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
# print(args.cuda)

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)

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
print(np.array(train_mat).shape)

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
model = ListMF(n_user, n_item+1, 
			   init_range=random_range, emb_size=emb_size).to(device)

# ---------------------------Train and test---------------------------------------
train_loss_all = []
test_loss_all = []
epoch_all = [] 
# leave an interface for the epochID incase we save the loss value larger than 1

# --------------------------Define optimizer----------------------------------
# best_result = 0
Best_ndcg = Best_ndcg_at_5 = Best_ap = Best_rr = Best_rbp_80 = Best_rbp_90 = Best_rbp_95 = 0
columns = ['loss_type', 'lr', 'threshold', 'reg', 'fold', 'frac', 'emb_size', 'NDCG@5', 'NDCG', 'RR', 'AP', 'RBP_80', 'RBP_90', 'RBP_95']
columns_indi = ['NDCG@5', 'NDCG', 'AP', 'RR', 'RBP_80', 'RBP_90', 'RBP_95']

if loss_type == 'rbp':
	weight_decay = 0
else:
	weight_decay = reg
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for i in tqdm(range(epochs)):
	# sum_loss = 0.0
	# if loss_type == 'rbp':
	# 	if (i + 1) % 20 == 0:
	# 		f_rbp *= 2
	# 		if f_rbp > 1:
	# 			f_rbp = 1
	train_loss = loss_train_test(model, optimizer, train_loader, loss_type, device, \
								t, b, temp, p, f_rbp, num_pos, num_neg)
	# print(train_loss.sum())

	# train_loss.sum().backward(retain_graph=True)
	# optimizer.step()
	# sum_loss += train_loss.sum()
	# print ("epoch %d" % i)
	# print("train loss %.4f" % sum_loss)
	# train_loss_all.append(sum_loss.cpu().tolist())
	# epoch_all.append(i)
	# print(torch.cuda.memory_summary())

	## ------------Evaluation------------------------
	if (i + 1) % 5 == 0:
		NDCG_at_5, NDCG, RR, AP, RBP_80, RBP_90, RBP_95, ndcg_at_5, ndcg, mrr, mAP, rbp_80, rbp_90, rbp_95 = evaluation(model, test_loader, max_rating, device, k, p, n_item)

		if loss_type == 'ap':
			if mAP > Best_ap:
				epoch = i
				Best_ap = mAP
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				print("Best" + args.loss_type.upper() + ": %.4f" % best_mAP)
				save_model(epoch, model, Best_ap, optimizer, save_path) 

				result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)

		elif loss_type == 'rr':
			if mrr > Best_rr:
				epoch = i
				Best_rr = mrr
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				print("Best" + args.loss_type.upper() + ": %.4f" % best_mrr)
				save_model(epoch, model, Best_rr, optimizer, save_path)

				result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False) 

		elif loss_type == 'rbp' or loss_type == 'nrbp' or loss_type == 'nrbp_1':
			if rbp_80 > Best_rbp_80:
				epoch = i
				Best_rbp_80 = rbp_80
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_80_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
			
				save_model(epoch, model, Best_rbp_80, optimizer, save_path) 

				loss_type_ = 'nrbp_80'
				print("Best" + loss_type_.upper() + ": %.4f" % best_rbp_80)

				result = pd.DataFrame([[loss_type_, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type_ + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)

			if rbp_90 > Best_rbp_90:
				epoch = i
				Best_rbp_90 = rbp_90
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_90_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				# print("Best" + args.loss_type.upper() + ": %.4f" % best_rbp_90)
				save_model(epoch, model, Best_rbp_90, optimizer, save_path) 

				loss_type_ = 'nrbp_90'
				print("Best" + loss_type_.upper() + ": %.4f" % best_rbp_90)

				result = pd.DataFrame([[loss_type_, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type_ + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)


			if rbp_95 > Best_rbp_95:
				epoch = i
				Best_rbp_95 = rbp_95
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_95_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				# print("Best" + args.loss_type.upper() + ": %.4f" % best_rbp_95)
				save_model(epoch, model, Best_rbp_95, optimizer, save_path) 

				loss_type_ = 'nrbp_95'
				print("Best" + loss_type_.upper() + ": %.4f" % best_rbp_95)

				result = pd.DataFrame([[loss_type_, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type_ + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)

		elif loss_type == 'dcg':
			if ndcg > Best_ndcg:
				epoch = i
				Best_ndcg = ndcg
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_80_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				print("Best" + args.loss_type.upper() + ": %.4f" % best_ndcg)
				save_model(epoch, model, Best_ndcg, optimizer, save_path) 

				result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)

			if ndcg_at_5 > Best_ndcg_at_5:
				epoch = i
				Best_ndcg_at_5 = ndcg_at_5
				best_ndcg = ndcg
				best_ndcg_at_5 = ndcg_at_5
				best_mrr = mrr
				best_mAP = mAP
				best_rbp_80 = rbp_80
				best_rbp_90 = rbp_90
				best_rbp_95 = rbp_95
				best_NDCG = NDCG
				best_NDCG_at_5 = NDCG_at_5
				best_RR = RR
				best_AP = AP
				best_RBP_80 = RBP_80
				best_RBP_90 = RBP_90
				best_RBP_95 = RBP_95
				model_file = "model_" + dataset + '_' + loss_type + '_80_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
				save_path = os.path.join(args.save_path, model_file)
				# print("Best" + args.loss_type.upper() + ": %.4f" % best_dcg)
				save_model(epoch, model, Best_ndcg_at_5, optimizer, save_path) 

				loss_type_ = 'ndcg_5'
				print("Best" + loss_type_.upper() + ": %.4f" % best_ndcg_at_5)

				result = pd.DataFrame([[loss_type_, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
				result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
				result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
				name = 'loss_type_' + loss_type_ + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
				res_path = os.path.join('./results/', dataset, 'overall')
				res_path_indi = os.path.join('./results/', dataset, 'individual')

				dir_exists(res_path)
				dir_exists(res_path_indi)
				result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
				result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)






# 		if loss_type == 'dcg':
# 			result_current = ndcg
# 		elif loss_type == 'rr':
# 			result_current = mrr
# 		elif loss_type == 'ap':
# 			result_current = mAP
# 		elif loss_type == 'rbp':
# 			result_current = rbp_80 + rbp_90 + rbp_95
# 		else:
# 			print('loss function does not exist.')

# 		if 

# 		if result_current > best_result:
# 			epoch = i
# 			best_result = result_current
# 			# best_hr = hr
# 			best_ndcg = ndcg
# 			best_ndcg_at_5 = ndcg_at_5
# 			best_mrr = mrr
# 			best_mAP = mAP
# 			best_rbp_80 = rbp_80
# 			best_rbp_90 = rbp_90
# 			best_rbp_95 = rbp_95
# 			best_NDCG = NDCG
# 			best_NDCG_at_5 = NDCG_at_5
# 			best_RR = RR
# 			best_AP = AP
# 			best_RBP_80 = RBP_80
# 			best_RBP_90 = RBP_90
# 			best_RBP_95 = RBP_95
# 			model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(frac) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
# 			save_path = os.path.join(args.save_path, model_file)
# 			print("Best" + args.loss_type.upper() + ": %.4f" % best_result)
# 			save_model(epoch, model, best_result, optimizer, save_path) 

# columns = ['loss_type', 'lr', 'threshold', 'reg', 'fold', 'frac', 'emb_size', 'NDCG@5', 'NDCG', 'RR', 'AP', 'RBP_80', 'RBP_90', 'RBP_95']
# result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)

# # print(result)
# columns_indi = ['NDCG@5', 'NDCG', 'AP', 'RR', 'RBP_80', 'RBP_90', 'RBP_95']
# result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
# result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
# # print(result_indi.head())
# name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
# res_path = os.path.join('./results/', dataset, 'overall')
# res_path_indi = os.path.join('./results/', dataset, 'individual')

# dir_exists(res_path)
# dir_exists(res_path_indi)

# result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
# result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)


# # df_res = pd.read_csv(res_path, header=0)
# # df_res = df_res.append(result, ignore_index=True)
# # df_res = df_res.round(4)
# # df_res.to_csv(res_path, index=False)

# # df_res_indi = pd.read_csv(res_path, header=0)
# # df_res_indi = df_res.append(result, ignore_index=True)
# # df_res_indi = df_res.round(4)
# # df_res_indi.to_csv(res_path, index=False)

	
# 	# test_loss = loss_train_test(model.eval(), optimizer, test_loader, loss_type, device, \
# 	#                             t, b, temp, p, f_rbp, num_neg, tradeoff, mode='test')

# 	# test_loss_all.append(test_loss.cpu().tolist())
# 	# epoch_all.append(i)



