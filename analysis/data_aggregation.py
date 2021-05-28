# -*- coding: utf-8 -*-
import os
import math
import argparse
import sys

import numpy as np 
import pandas as pd 

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--res_path', nargs='?', default='../results/',
						help='Input data path.') # change to ../lambda_results for pairwise files
parser.add_argument('--dataset', type=str, default='Epinions',
					choices=['Epinions', 'citeulike', 'ml-10m', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Sports_and_Outdoors']) 
parser.add_argument('--p', type=float, default=0.95)
args = parser.parse_args() 

res_path = args.res_path
dataset = args.dataset
p = args.p

data_path = os.path.join(res_path, dataset, 'overall')
# data_path = os.path.join(res_path, dataset, 'overall', str(p))
results = os.listdir(data_path)

for item in results:
	if 'rbp_0.8' in item:
		df = pd.read_csv(os.path.join(data_path, item))
		df.loss_type = 'lambda_rbp_80'
		df.to_csv(os.path.join(data_path, item), index=False)
	if 'rbp_0.9' in item:
		df = pd.read_csv(os.path.join(data_path, item))
		df.loss_type = 'lambda_rbp_90'
		df.to_csv(os.path.join(data_path, item), index=False)
	if 'rbp_0.95' in item:
		df = pd.read_csv(os.path.join(data_path, item))
		df.loss_type = 'lambda_rbp_95'
		df.to_csv(os.path.join(data_path, item), index=False)


df = pd.read_csv(os.path.join(data_path, results[0]), index_col=None)


for i in range(1, len(results)):
	df_ = pd.read_csv(os.path.join(data_path, results[i]))
	df = pd.concat([df, df_], axis=0)


df.to_csv(os.path.join(os.path.dirname(data_path), 'all_res.csv'), index=False)



