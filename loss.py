import torch
use_gpu = torch.cuda.is_available()

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, targets):
        D = embeddings.size(0)
        S = embeddings.size(1)
        anchor = embeddings[:, targets.data == -1].permute(1, 0).repeat(1, S-2).view((S-2), D).permute(1, 0)
        positive = embeddings[:, targets.data == 1].permute(1, 0).repeat(1, S-2).view((S-2), D).permute(1, 0)
        negative = embeddings[:, targets.data == 0]

        dist_pos = torch.sum(torch.pow(anchor-positive, 2), dim=0)
        dist_neg = torch.sum(torch.pow(anchor-negative, 2), dim=0)

        return torch.sum(torch.clamp(dist_pos - dist_neg + self.margin, min=0))
