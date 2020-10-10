import torch
from torchvision import models

class VGG(torch.nn.Module):
    def __init__(self, mode, pretrained=True):
        super(VGG, self).__init__()
        if mode == 'vgg16':
            self.features = models.vgg16_bn(pretrained=pretrained)
        elif mode == 'vgg19':
            self.features = models.vgg19_bn(pretrained=pretrained)
    
    def forward(self, x):
        x = self.features.features(x)
        return x
