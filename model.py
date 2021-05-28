import torch
import torch.nn as nn

class ListMF(nn.Module):
    def __init__(self, n_user, n_item, init_range, emb_size, 
    			weight_user=None, 
    			weight_item=None):

        super(ListMF, self).__init__()
        self.user_emb = nn.Embedding(n_user, emb_size)
        self.item_emb = nn.Embedding(n_item, emb_size)
          # initlializing weights
        if weight_user == None:
            self.user_emb.weight.data.uniform_(-init_range, init_range)
        else:
            self.user_emb = weight_user
           
        if weight_item == None:
            self.item_emb.weight.data.uniform_(-init_range, init_range)
        else:
            self.item_emb = weight_item
        
    def forward(self, userID, itemID, rels, mode):
        # score = torch.empty(size=(userID.size()[0], items.size()[1]))
        user = self.user_emb(userID)
        items = self.item_emb(itemID)

        if mode == 'train':
            pred = (user * items).sum(-1)
            idx_pad = (rels == 20).nonzero()
            pred[idx_pad[:, 0], idx_pad[:, 1]] = -100
            rels[idx_pad[:, 0], idx_pad[:, 1]] = 0
            return pred, rels
        
        return (user * items).sum(-1)
        # return pred