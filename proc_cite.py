import pandas as pd
import lenskit.crossfold as xf
import numpy as np
import json
from utils import *

filepath = 'data/citeulike/users.dat'

with open(filepath, 'r') as f:
	lines = [line.strip().split(' ') for line in f]

lines = [list(map(int, line)) for line in lines]

max_id = 0
length = 0

for line in lines:
	length += len(line) - 1
	if max(line) > max_id:
		max_id = max(line)

n_user = len(lines)
n_item = max_id + 1
print(n_user, n_item, length)

triplet = []
for i in range(n_user):
	for j in range(1, len(lines[i])):
		triplet.append([i, lines[i][j], 1])

columns = ['user', 'item', 'rating']
ratings = pd.DataFrame(triplet, columns=columns)

df_25 = ratings[ratings.user.isin(ratings.user.value_counts()[ratings.user.value_counts() >= 25].index)]
df_25 = df_25.reset_index(drop=True)
print(len(pd.unique(df_25.user)), len(pd.unique(df_25.item)), len(df_25))

_, df_25 = get_unique_id(df_25, 'user')
_, df_25 = get_unique_id(df_25, 'item')

n_user = df_25.user_id.drop_duplicates().size
n_item = df_25.item_id.drop_duplicates().size
print(n_user, n_item)


dataset_meta_info_0 = {'dataset_size': len(df_25),
                     'user_size': n_user,
                     'item_size': n_item
                     }
with open(os.path.join('data', 'citeulike', 'th_0', 'dataset_meta_info_0.json'), 'w') as f:
	json.dump(dataset_meta_info_0, f)                 

seeds = [1, 777, 1992, 2003, 2020]
# df_30_train = 
for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = os.path.join('data', 'citeulike', 'th_0', 'fold_'+str(j))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))

