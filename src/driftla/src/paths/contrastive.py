
import torch
import torch.nn.functional as F


def intra_path_contrastive_loss(target_emb, positive_embs, negative_embs, tau=0.05):
    if positive_embs.shape[0] == 0:
        return torch.tensor(0.0, device=target_emb.device)

    pos_logits = torch.mv(positive_embs, target_emb) / tau
    neg_logits = torch.mv(negative_embs, target_emb) / tau


    log_num = torch.logsumexp(pos_logits, dim=0)

    log_den = torch.logsumexp(torch.cat([pos_logits, neg_logits]), dim=0)

    loss = -(log_num - log_den)
    return loss


def inter_path_contrastive_loss(center_emb, positive_embs, negative_embs, tau=0.05):
    if positive_embs.shape[0] == 0:
        return torch.tensor(0.0, device=center_emb.device)

    pos_logits = torch.mv(positive_embs, center_emb) / tau
    neg_logits = torch.mv(negative_embs, center_emb) / tau

    log_num = torch.logsumexp(pos_logits, dim=0)
    log_den = torch.logsumexp(torch.cat([pos_logits, neg_logits]), dim=0)

    loss = -(log_num - log_den)
    return loss


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau

    def forward(self, target_emb, positive_embs, negative_embs):
        return intra_path_contrastive_loss(target_emb, positive_embs, negative_embs, self.tau)

    def inter_path_loss(self, center_emb, positive_embs, negative_embs):
        return inter_path_contrastive_loss(center_emb, positive_embs, negative_embs, self.tau)
