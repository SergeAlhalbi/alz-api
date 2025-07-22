import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2MRI(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.model = mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)