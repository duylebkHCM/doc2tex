import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, reduction, classes, ignore_index, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = ignore_index
        self.reduction = reduction

    def forward(self, pred: torch.FloatTensor, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        loss = torch.sum(-true_dist * pred, dim=self.dim)
        if not self.reduction:
            return torch.mean(loss)
        else:
            return loss
