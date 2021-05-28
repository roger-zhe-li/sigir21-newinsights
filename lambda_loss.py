import torch
import numpy as numpy
import torch.nn as nn
import torchsnooper



class lambda_loss:
	def __init__(self, device, rank_list, target_list, t, b, p, num_pos, num_neg):              
		# self.rank_list = rank_list.unsqueeze(1)
		# self.target_list = target_list.unsqueeze(1)
		self.rank_list = rank_list
		self.target_list = target_list
		self.num_pos = num_pos
		self.num_neg = num_neg
		self.device = device
		self.p = p

	
	def dcg(self, rank_list, target_list):
		return torch.sum((torch.pow(2, target_list) - 1)/ torch.log2(2 + rank_list.float()), dim=1)
	
	def rbp(self, rank_list, target_list, p):
		return torch.sum(target_list * torch.pow(p, rank_list), dim=1)

	def rr(self, rank_list, target_list):
		rank_list = rank_list + 1
		rr_all = target_list / rank_list
		values, _ = torch.max(rr_all, dim=1)
		# print(values)
		return values
	
	def smart_sort(self, x, permutation):
		d1, d2 = x.size()
		ret = x[
			torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
			permutation.flatten()
		].view(d1, d2)
		return ret
	

	def ap(self, rank_list, target_list):
		value, idxs = torch.sort(rank_list)        
		target_reorder = self.smart_sort(target_list, idxs)
		rank_list = value + 1
		ap_ind = target_reorder * target_reorder.cumsum(dim=1) / rank_list
		# print((ap_ind != 0).sum(dim=1))
		ap = ap_ind.sum(1) / (ap_ind != 0).sum(dim=1)
		return ap

	# @torchsnooper.snoop()
	def lambda_dcg(self):
		device = self.device
		rank_list = self.rank_list
		target_list = self.target_list

		num_doc, n_docs = rank_list.size()

		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)

		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)
		# doc_ranks = doc_ranks.permute(0, 2, 1)

		# print(rank_list[:, :n_rel].size())
		# print(rank_list[:, n_rel:].size())

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)
		
	
		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 

		N = 1
 
		dcg_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			diff_new = 1 / (1 + doc_ranks[i, :, :rel]).log2().permute(1, 0) - (1 / (1 + doc_ranks[i, :, rel:val]).log2())
			# norm = (1 / (2 + torch.arange(rel).float()).log2()).sum().to(device)
			# diff_new = diff_new / norm
			# print(n_docs-rel)
			# print(n_docs+rel-val)
			dcg_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(diff_new)

		lamb_updates = 1 / (1 + exped) * N * dcg_diffs.abs()
		loss = lamb_updates.sum()
	   
		return loss


	def lambda_ap(self):
		rank_list = self.rank_list
		target_list = self.target_list

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			# print(n_rel)
			# print(n_val)
		# print(n_rel.shape)

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)

		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)

		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 
	

		N = 1
		ap_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]

			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			# print(rel, val)

			rank_new = torch.zeros([rel, val-rel]).to(device)


			for j in range(rel):
				rank_p = doc_ranks[i, :, j].item()
				
				# print(1.0 * (doc_ranks[i] <= rank_p) * 1.0 * (target_list[i] == 1))
				# target_list[i, :val] == 1
				
				m = (1.0 * (doc_ranks[i] <= rank_p) * 1.0 * (target_list[i] == 1)).sum(-1)

				# m = (1.0 * ((target_list[i, :val] == 1) and (rank_list[i] <= rank_p))).sum(-1)
				term_2 = (m / rank_p).item()

				for k in range(val-rel):
					rank_n = (doc_ranks[i, :, val-rel+k]).item()
					
					if rank_p < rank_n:
						rank_new[j, k] = 0
					else:
						n = (1.0 * (doc_ranks[i] <= rank_n) * 1.0 * (target_list[i] == 1)).sum(-1)
						term_1 = (n + 1) / rank_n
						term_1 = term_1.item()

						prec = target_list[i, : val] * doc_ranks[i, :, :val]
						prec = prec.squeeze(0).double()
						# print(prec)
						# print(rank_p, rank_n)
			
						
					
						prec = torch.where(prec > rank_n, prec, 0.)
						prec = torch.where(prec < rank_p, prec, 0.)

						prec = prec[prec.nonzero()]

						if prec == torch.Size([]):
							term_3 = 0
						else:
							term_3 = (1.0 / prec).sum()
						# print(term_3)
						

						rank_new[j, k] = (term_1 - term_2 + term_3) / rel
				ap_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(rank_new) 

		lamb_updates = 1 / (1 + exped) * N * ap_diffs.abs()
		loss = lamb_updates.sum()
		return loss


	def lambda_rr(self):
		# device = self.device
		rank_list = self.rank_list
		target_list = self.target_list
		# p = self.p

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)
		
	
		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 

		rr_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]

			rank_new = torch.zeros([rel, val-rel]).to(device)

			diff_new_ = 1 / doc_ranks[i, :, :rel].permute(1, 0) - 1 / doc_ranks[i, :, rel:val]
			diff_new_ = torch.clamp(diff_new_, max=0)
			top_rel = torch.argmin(doc_ranks[i, :, :rel])
			diff_new = torch.zeros_like(diff_new_)
			diff_new[top_rel] = diff_new_[top_rel]

			rr_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(diff_new)
		
		N = 1
		lamb_updates = 1 / (1 + exped) * N * rr_diffs.abs()
		loss = lamb_updates.sum()
		return loss


	def lambda_rbp(self):
		device = self.device
		rank_list = self.rank_list
		target_list = self.target_list
		p = self.p

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			# print(n_rel)
			# print(n_val)
		# print(n_rel.shape)

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)
		# doc_ranks = doc_ranks.permute(0, 2, 1)

		# print(rank_list[:, :n_rel].size())
		# print(rank_list[:, n_rel:].size())

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)