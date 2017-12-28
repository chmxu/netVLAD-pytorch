import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropy(nn.Module):
    def __init__(self, epsilon=10e-6, num_classes=239):
        super(CrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.eps = epsilon

    def forward(self, logits, labels):
        float_labels = Variable(torch.zeros(labels.shape[0], self.num_classes).scatter_(1, labels.data.cpu().view(-1, 1), 1.0))
        if torch.cuda.is_available():
            float_labels = float_labels.cuda()
        cross_entropy_loss = float_labels * torch.log(logits + self.eps) + (
          1 - float_labels) * torch.log(1 - logits + self.eps)
        cross_entropy_loss = -1 * cross_entropy_loss
        return torch.mean(torch.sum(cross_entropy_loss, 1))
