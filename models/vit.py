import torch.nn as nn
from torchvision.models import vit_b_16

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.base = vit_b_16(weights=weights)
        self.base.heads[0] = nn.Linear(self.base.heads[0].in_features, num_classes)

    def forward(self, x):
        return self.base(x)