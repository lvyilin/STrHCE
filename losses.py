import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def __init__(self, epsilon=0.05):
        super(HingeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_main, input_aux, target):
        target = target.unsqueeze(1)
        p_main = torch.gather(input_main, 1, target)
        p_aux = torch.gather(input_aux, 1, target)
        return F.relu(p_main - p_aux + self.epsilon).mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
