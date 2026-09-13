from torch import nn


class BFOR_Loss(nn.Module):
    def __init__(self, alpha, lambda_ctr, k):
        self.alpha = alpha
        self.lambda_ctr = lambda_ctr
        self.k = k

    # preds should be a tensor of B dictionaries
    # target should be a list of tensors of shape [N_box, 4]
    def forward(self, preds, targets):
        pass

    def get_scale(self, box):
        pass
