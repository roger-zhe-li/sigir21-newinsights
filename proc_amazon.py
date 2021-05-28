import pandas as pd
import lenskit.crossfold as xf
import numpy as np 
from utils import *
import json

ratings = pd.read_csv('data/Clothing_Shoes_and_Jewelry/Home_and_Kitchen.csv', header=None, index_col=None)
#

dir_exists('data/Clothing_Shoes_and_Jewelry/th_0')
dir_exists('data/Clothing_Shoes_and_Jewelry/th_4')
dir_exists('data/Clothing_Shoes_and_Jewelry/th_5')

columns = ['item', 'user', 'rating', 'timestamp']
ratings.columns = columns
ratings = ratings[['user', 'item', 'rating', 'timestamp']]
ratings = ratings.drop('timestamp', axis=1)


n_user = len(pd.unique(ratings.user))
n_item = len(pd.unique(ratings.item))
print(n_user, n_item, len(ratings))

df_25 = ratings[ratings.user.isin(ratings.user.value_counts()[ratings.user.value_counts() >= 25].index)]
df_25 = df_25.reset_index(drop=True)
print(len(pd.unique(df_25.user)))

_, df_25 = get_unique_id(df_25, 'user')
_, df_25 = get_unique_id(df_25, 'item')
print(df_25.head())

n_user = df_25.user_id.drop_duplicates().size
n_item = df_25.item_id.drop_duplicates().size
print(n_user, n_item)

df_th_4 = ratings.loc[ratings.rating >= 4]
df_th_5 = ratings.loc[ratings.rating >= 5]

df_25_th_4 = df_th_4[df_th_4.user.isin(df_th_4.user.value_counts()[df_th_4.user.value_counts() >= 25].index)]
df_25_th_5 = df_th_5[df_th_5.user.isin(df_th_5.user.value_counts()[df_th_5.user.value_counts() >= 25].index)]
df_25_th_4 = df_25_th_4.reset_index(drop=True)
df_25_th_5 = df_25_th_5.reset_index(drop=True)

_, df_25_th_4 = get_unique_id(df_25_th_4, 'user')
_, df_25_th_4 = get_unique_id(df_25_th_4, 'item')

_, df_25_th_5 = get_unique_id(df_25_th_5, 'user')
_, df_25_th_5 = get_unique_id(df_25_th_5, 'item')

print(len(pd.unique(df_25.user)), len(pd.unique(df_25_th_4.user)), len(pd.unique(df_25_th_5.user)))
print(len(pd.unique(df_25.item)), len(pd.unique(df_25_th_4.item)), len(pd.unique(df_25_th_5.item)))
print(len(df_25), len(df_25_th_4), len(df_25_th_5))
# print(df_25_th_5.head())
dataset_meta_info_0 = {'dataset_size': len(df_25),
                     'user_size': n_user,
                     'item_size': n_item
                     }
dataset_meta_info_4 = {'dataset_size': len(df_25_th_4),
                     'user_size': len(pd.unique(df_25_th_4.user)),
                     'item_size': len(pd.unique(df_25_th_4.item))
                     }    
dataset_meta_info_5 = {'dataset_size': len(df_25_th_5),
                     'user_size': len(pd.unique(df_25_th_5.user)),
                     'item_size': len(pd.unique(df_25_th_5.item))
                     }                                    
                       

with open(os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_0', 'dataset_meta_info_0.json'), 'w') as f:
	json.dump(dataset_meta_info_0, f)
with open(os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_4', 'dataset_meta_info_4.json'), 'w') as f:
	json.dump(dataset_meta_info_4, f)
with open(os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_5', 'dataset_meta_info_5.json'), 'w') as f:
	json.dump(dataset_meta_info_5, f)

seeds = [1, 777, 1992, 2003, 2020]
# df_25__train = 
for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_0', 'fold_'+str(j))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))


for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25_th_4, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_4', 'fold_'+str(j))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))


for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25_th_5, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = os.path.join('data', 'Clothing_Shoes_and_Jewelry', 'th_5', 'fold_'+str(j))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))