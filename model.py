import torch
from resnet import Resnet

class TripletNet(torch.nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.backbone = Resnet(mode='resnet34')
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=128, bias=False)
        self.dropout = torch.nn.Dropout(p=0.2)

    def _embedding_net(self, x):
        feature = self.backbone(x)
        feature = self.pool(feature).reshape(feature.size(0), -1)
        if self.training:
            feature = self.dropout(feature)
        feature = self.fc(feature)
        feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
        return feature

    def forward(self, x):
        return self._embedding_net(x)
