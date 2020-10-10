import torch
from torchvision import models

class Resnet(torch.nn.Module):
    def __init__(self, mode, pretrained=True):
        super(Resnet, self).__init__()
        if mode == 'resnet18':
            self.features = models.resnet18(pretrained=pretrained)
        elif mode == 'resnet34':
            self.features = models.resnet34(pretrained=pretrained)
        elif mode == 'resnet50':
            self.features = models.resnet50(pretrained=pretrained)
        elif mode == 'resnet101':
            self.features = models.resnet101(pretrained=pretrained)

        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.max_pool = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.max_pool(x)
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        return feature4