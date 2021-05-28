import os
import torch
from torch import nn
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
from list_loss import *
from lambda_loss import lambda_loss

# make folders for data, models and results
def dir_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_model(epoch, model, best_result, optimizer, save_path):
	torch.save({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_performance': best_result,
		'optimizer': optimizer.state_dict(),
		}, save_path)

# ---------------------------Amazon Dataset Preprocessing Functions---------------

def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
	"""
	clear the ids
	:param data_pd: pd.DataFrame 
	:param column: specified col
	:return: dict: {value: id}
	"""
	new_column = '{}_id'.format(column)
	assert new_column not in data_pd.columns
	temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
	temp[new_column] = temp.index
	temp.index = temp[column]
	del temp[column]
	# data_pd.merge()
	data_pd = pd.merge(left=data_pd,
		right=temp,
		left_on=column,
		right_index=True,
		how='left')

	return temp[new_column].to_dict(), data_pd


def load_data_amazon(df):
	user_ids, data_pd = get_unique_id(df, 'user')
	item_ids, data_pd = get_unique_id(df, 'item')


	data_pd = data_pd.loc[:, ['user', 'user_id', 'item', 'item_id', 'rating', 'timestamp']]
	data_pd = data_pd.drop(['user', 'item'], axis=1)

	return data_pd


# -----------------train and test loss plot----------------------------
# def plot_figure(loss_type, dataset, lr, temp, num_neg, threshold, tradeoff, k, fig_path_, train_loss, test_loss):
# 	curve_file = "curve_" + loss_type + '_' + str(lr) + '_' + str(temp) + '_' + str(num_neg) + '_' + str(threshold) + '_' + str(tradeoff) + '_k_' + str(k) + '.png'
# 	dir_exists(os.path.join(fig_path_, dataset))
# 	fig_path = os.path.join(fig_path_, dataset, curve_file)

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	ax.plot(train_loss, '-', label='train loss')

# 	ax2 = ax.twinx()
# 	ax2.plot(test_loss, '-r', label='test loss')
# 	fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

# 	ax.set_xlabel('epoch')
# 	ax.set_ylabel('training loss')
# 	ax2.set_ylabel('test loss')

# 	plt.axis('off')
# 	plt.savefig(fig_path)

def write_result(res_path, dataset, loss_type, num_neg, lr, temp, threshold, tradeoff, k, hr, ndcg, mrr, mAP, rbp):
	dir_exists(res_path)
	res_path = os.path.join(res_path, str(dataset) + '.txt')
	f = open(res_path, 'a+')
	f.write('loss type: ' + loss_type + ', num_neg: ' + str(num_neg) + ', learning rate: ' + str(lr) + ', temperature: ' + str(temp) + ', threshold: ' + str(threshold) + ', tradeoff:' + str(tradeoff) + ', k:' + str(k) + '\n')
	f.write('HR = ' + str(hr) + ', NDCG = ' + str(ndcg) + ', MRR = ' + str(mrr) + ', MAP = ' + str(mAP) + ', RBP = ' + str(rbp) + '\n\n' )
	f.close()


def choose_loss(loss_type, device, prediction, rel, t, b, temp, p, f_rbp, num_pos, num_neg):
	if loss_type == 'dcg':
		loss_value = ndcg_loss(device, prediction, rel, t, b, num_pos, num_neg, temp).to(device)
	elif loss_type == 'rr':
		loss_value = rr_loss(device, prediction, rel, temp).to(device)
	elif loss_type == 'ap':
		loss_value = ap_loss(device, prediction, rel, temp).to(device)
	elif loss_type == 'rbp':
		loss_value = rbp_loss(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'nrbp':
		loss_value = nrbp_loss(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'nrbp_1':
		loss_value = nrbp_loss_1(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'lambda_dcg':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_dcg().to(device)
	elif loss_type == 'lambda_rr':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_rr().to(device)
	elif loss_type == 'lambda_ap':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_ap().to(device)
	elif loss_type == 'lambda_rbp':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_rbp().to(device)

	else: print('The loss function of your choice is not available.')

	return loss_value

def loss_train_test(model, optimizer, dataloader, loss_type, device, t, b, temp, p, f_rbp, num_pos, num_neg):
	loss = 0
	# if mode == 'train':
	for user, items, rels in dataloader:
		batch = len(user)
		user, items, rel = user.to(device), items.to(device), rels.type(torch.FloatTensor).to(device)       
		user = torch.unsqueeze(user, 1)  

		prediction, rel = model(user, items, rel, mode='train')
		# idx_pad = (rel == 20).nonzero()
		# print(prediction.size())
		# print(rels.size())
		loss_value = choose_loss(loss_type, device, \
			prediction, rel, t, b, temp, p, f_rbp, num_pos, num_neg)
		# for param in model.parameters():
		# 	regularization_loss += torch.sum(torch.abs(param))
		# loss_value += 0.0001 * regularization_loss
		# print(loss_value)
		# print(loss_value / user.size(0))
		optimizer.zero_grad()
		loss_value.sum().backward()
		optimizer.step()
		loss += loss_value.sum()
	# elif mode == 'test':
	# 	for user, items, rels, _ in dataloader:
	# 		batch = len(user)
	# 		user, items, rel = user.to(device), items.to(device), rels.type(torch.FloatTensor).to(device)       
	# 		user = torch.unsqueeze(user, 1)  

	# 		prediction = model(user, items)
	# 		loss_value = choose_loss(loss_type, device, \
	# 			prediction, rel, t, b, temp, p, f_rbp, num_neg, tradeoff)
	# 		optimizer.zero_grad()
	# 		loss_value.sum().backward()
	# 		optimizer.step()
	# 		loss += loss_value.sum()
	return loss
