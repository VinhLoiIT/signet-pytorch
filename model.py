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

class Encoder(nn.Module):
    def __init__(self, feature_map=False):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 96, 11),  # size = [B, 1, 145, 210]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [B, 96, 72, 105]
            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [B, 256, 72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [B, 256, 36, 52]
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),  # size = [B, 256, 18, 26]
            nn.Dropout2d(p=0.3),
        )
        if feature_map:
            self.out = nn.Identity()
        else:
            self.out = nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(256 * 18 * 26, 1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 128),
            )

    def forward(self, x):
        return self.out(self.encoder(x))


class SigNet(nn.Module):
    def __init__(self, glorot_uniform=True):
        super(SigNet, self).__init__()
        self.encoder = Encoder(feature_map=False)
        if glorot_uniform:
            glorot_init_uniform(self.encoder)

    def forward(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)


def glorot_init_uniform(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    B, C, H, W = 1, 1, 155, 220
    data1 = torch.randn((B, C, H, W))
    data2 = torch.randn((B, C, H, W))
    encoder = Encoder(feature_map=False)
    features = encoder(data1)
    assert list(features.shape) == [1, 128]
    print('Check: encoder without feature_map, passed.')
    encoder_with_feature_map = Encoder(feature_map=True)
    feature_map = encoder_with_feature_map(data1)
    assert list(feature_map.shape) == [B, 256, 18, 26]
    print('Check: encoder with feature_map, passed.')
    net = SigNet()
    out = net(data1, data2)
    assert len(out) == 2
    print('Check: num of output for siamese net, passed.')
    y1, y2 = out
    assert list(y1.shape) == [B, 128] and list(y2.shape) == [B, 128]
    print('Check: shape of outputs for siamese net, passed.')
    print('All checks passed.')
