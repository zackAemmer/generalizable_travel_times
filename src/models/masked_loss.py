from torch import nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        loss = self.mse_loss(input, target)
        loss = (loss * mask.float()).sum()
        num_non_padded = mask.float().sum()
        loss = loss / num_non_padded
        return loss