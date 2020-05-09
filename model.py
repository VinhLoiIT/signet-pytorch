import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, x1, x2, y):
        '''
        Shapes:
        -------
        x1: [B,C]
        x2: [B,C]
        y: [B,1]

        Returns:
        --------
        loss: [B,1]]
        '''
        distance = torch.pairwise_distance(x1, x2, p=2)
        loss = self.alpha * (1-y) * distance**2 + \
               self.beta * y * (torch.max(torch.zeros_like(distance), self.margin - distance)**2)
        return torch.mean(loss, dtype=torch.float)

class SigNet(nn.Module):
    '''
    Reference Keras: https://github.com/sounakdey/SigNet/blob/master/SigNet_v1.py
    '''
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #input size = [155, 220, 1]
            nn.Conv2d(1, 96, 11), # size = [145,210]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2), # size = [72, 105]
            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'), # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2), # size = [36, 52]
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2), # size = [18, 26]
            nn.Dropout2d(p=0.3),
            nn.Flatten(1, -1), # 18*26*256
            nn.Linear(18*26*256, 1024),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 256),
        )

        # TODO: init bias = 0

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        return x1, x2
