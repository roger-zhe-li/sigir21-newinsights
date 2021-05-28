class lambda_loss:
    def __init__(self,
                    rank_list,
                    target_list,
                    max_scale=5,                
                    ):
            # t: gain index for dcg
            # b: discount index for dcg
            # k: cut-off value, by default set at 5 for dcg@5
            # rank_list: rank list
            # target_list: ground-truth     
            self.rank_list = rank_list
            self.target_list = target_list
            # self.rank_list = rank_list
            # self.target_list = target_list
            self.max_scale = max_scale

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