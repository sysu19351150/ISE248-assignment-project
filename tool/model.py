import torch
import torch.nn as nn
from torchvision import models

# 模型部分，使用resnet101架构
class SequencesNet(nn.Module):
    def __init__(self):
        super(SequencesNet, self).__init__()

        model = models.resnet101(pretrained=False)
        pre = torch.load("model/resnet101-5d3b4d8f.pth")
        model.load_state_dict(pre)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        channel_in = model.fc.in_features
        numClasses = 5
        model.fc = nn.Linear(channel_in, numClasses)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


