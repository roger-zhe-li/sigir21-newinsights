import pandas as pd
import lenskit.crossfold as xf
import numpy as np
from utils import *
import json


ratings = pd.read_csv('data/Epinions/trust_data.txt', header=None, index_col=None, sep=' ')
ratings.dropna(axis=1, how='all', inplace=True) 

columns = ['user', 'item', 'rating']
ratings.columns = columns
print(ratings.head())


n_user = len(pd.unique(ratings.user))
n_item = len(pd.unique(ratings.item))
print(n_user, n_item, len(ratings))


df_25 = ratings[ratings.user.isin(ratings.user.value_counts()[ratings.user.value_counts() >= 25].index)]
df_25 = df_25.reset_index(drop=True)
print(len(pd.unique(df_25.user)), len(pd.unique(df_25.item)), len(df_25))
print(df_25.head())

_, df_25 = get_unique_id(df_25, 'user')
_, df_25 = get_unique_id(df_25, 'item')
print(df_25.head())

n_user = df_25.user_id.drop_duplicates().size
n_item = df_25.item_id.drop_duplicates().size
print(n_user, n_item)

dataset_meta_info = {'dataset_size': len(df_25),
                     'user_size': n_user,
                     'item_size': n_item
                     }
                       

with open(os.path.join('data', 'Epinions', 'th_0', 'dataset_meta_info_0.json'), 'w') as f:
	json.dump(dataset_meta_info, f)



seeds = [1, 777, 1992, 2003, 2020]
# df_30_train = 
for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = os.path.join('data', 'Epinions', 'th_0', 'fold_'+str(j))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))

