import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.base.heads[0] = nn.Linear(self.base.heads[0].in_features, num_classes)

    def forward(self, x):
        return self.base(x)