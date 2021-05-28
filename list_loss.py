import torch
import numpy as np
import torchsnooper

def ndcg_loss(device, rank_list, target_list, t, b, num_pos, num_neg, temp=1.0):
    # t: gain index for dcg
    # b: discount index for dcg
    # getting rank for each item
    size = rank_list.size()[-1]
    rank_list = torch.unsqueeze(rank_list, 1)
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) - 0.5)
    numerator = torch.pow(t, target_list) - 1.0
    denominator = torch.log(rank_overall + b) / torch.log(torch.tensor(float(b)))

    len_rank = num_neg + num_pos

    sorted_target, indices = target_list.sort(descending=True)
    idcg_nume = torch.pow(t, sorted_target) - 1.0  
    idcg_deno = torch.log(1.0 * b + torch.arange(size)) / torch.log(torch.tensor(float(b)))
    idcg = (idcg_nume / idcg_deno.to(device)).sum(-1) + 1e-8
    score = (numerator / denominator).sum(-1)
    score_norm = score / idcg
    loss = -score_norm.sum()
    return loss


# @torchsnooper.snoop()
def rr_loss(device, rank_list, target_list, temp=1.0):
    # getting rank for each item
    length = len(rank_list)

    size = rank_list.size()[-1]
    target_list = torch.unsqueeze(target_list, 1)
    rank_list = torch.unsqueeze(rank_list, 1)
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
    rank_overall = torch.unsqueeze(rank_overall, 1) 
    factor_1 = target_list / rank_overall

    rank_overall_t = rank_overall.permute(0, 2, 1) # rank for all items using broadcast

    pairwise_ranking = torch.sigmoid(temp * (rank_overall_t - rank_overall))

    bias_mat = (0.5 * torch.eye(size)).unsqueeze(0).to(device)
    bias_mat = bias_mat.repeat(length, 1, 1)
    pairwise_ranking = pairwise_ranking - bias_mat
    factor_2 = torch.prod(1 - target_list * pairwise_ranking, -1)

    rr_value = (factor_1.squeeze(1) * factor_2).sum(1)
    rr_loss = -rr_value.sum()       
    return rr_loss

# # @torchsnooper.snoop()
# def rr_loss(device, rank_list, target_list, temp=1.0):
#     # getting rank for each item
#     length = len(rank_list)

#     size = rank_list.size()[-1]
#     factor_1 = torch.log(torch.sigmoid(rank_list))
#     # target_list = torch.unsqueeze(target_list, 1)
#     rank_list = torch.unsqueeze(rank_list, 1)

#     rank_list_t = rank_list.permute(0, 2, 1)
#     rank = rank_list - rank_list_t

#     factor_2 = torch.log(1 - target_list.unsqueeze(1) * torch.sigmoid(rank))

#     rr_value = target_list * (factor_1 * factor_2.squeeze(1).sum(1))
#     rr_loss = -rr_value.sum(1)       
#     return rr_loss


# @torchsnooper.snoop()
# def rbp_loss(device, rank_list, target_list, p, f_rbp, temp=1.0):

#     size = rank_list.size()[-1]
#     rank_list = torch.unsqueeze(rank_list, 1)
#     rank_list_t = rank_list.permute(0, 2, 1)
#     rank = rank_list_t - rank_list

#     rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
#     decay = torch.pow(p, f_rbp * rank_overall - 1)

#     rbp = (1 - p) * ((target_list * decay).sum())
#     rbp_loss = -rbp.sum()

#     return rbp_loss

# @torchsnooper.snoop()
def rbp_loss(device, rank_list, target_list, p, f_rbp, temp=1.0):
    size = rank_list.size()[-1]
    rank_list = torch.unsqueeze(rank_list, 1)
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
    rbp_loss = (target_list * (rank_overall - 1)).sum()
    target_list_sum = target_list.sum(-1)
    
    norm_factor = 0.5 * (target_list_sum - 1) * (target_list_sum - 2)
    rbp_loss = rbp_loss - norm_factor.sum()
    return rbp_loss


def nrbp_loss(device, rank_list, target_list, p, f_rbp, temp=1.0):
    size = rank_list.size()[-1]
    rank_list = torch.unsqueeze(rank_list, 1)
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
    rbp_loss = (target_list * (rank_overall - 1)).sum()
    target_list_sum = target_list.sum(-1)
    
    norm_factor = 0.5 * (target_list_sum - 1) * (target_list_sum - 2)
    nrbp_loss = -norm_factor.sum() / rbp_loss
    return nrbp_loss

# @torchsnooper.snoop()
def nrbp_loss_1(device, rank_list, target_list, p, f_rbp, frac=1.0, temp=1.0):
    size = rank_list.size()[-1]
    rank_list = torch.unsqueeze(rank_list, 1)
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
    rbp_loss = (target_list * (rank_overall - 1)).sum(-1)
    target_list_sum = target_list.sum(-1) * frac
    rbp_loss = (rbp_loss / target_list_sum).sum()

    return rbp_loss



def ap_loss(device, rank_list, target_list, temp=1.0):

    length = len(rank_list)

    rank_list = torch.unsqueeze(rank_list, 1)
    target_list = torch.unsqueeze(target_list, 1)
    size = rank_list.size()[-1]
    rank_list_t = rank_list.permute(0, 2, 1)
    rank = rank_list_t - rank_list

    rank_overall = (torch.sigmoid(temp * rank).sum(1) + 0.5)
    rank_overall = torch.unsqueeze(rank_overall, 1)        

    rank_overall_t = rank_overall.permute(0, 2, 1)

    pairwise_ranking = torch.sigmoid(temp * (rank_overall_t - rank_overall))

    bias_mat = (0.5 * torch.eye(size)).unsqueeze(0).to(device)
    bias_mat = bias_mat.repeat(length, 1, 1)
    ap_matrix = pairwise_ranking + bias_mat

    target_list_mat = target_list.repeat(1, size, 1)

    factor_1 = (target_list_mat * ap_matrix).sum(1)
    factor_2 = (target_list / rank_overall).squeeze(1)
    factor_3 = (torch.tensor(1).float().to(device) / target_list.sum())

    ap = factor_3 * ((factor_2 * factor_1).sum())

    loss = -ap

    return loss