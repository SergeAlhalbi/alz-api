import torch
from torch import nn
import torch.nn.functional as F
from models.vit import ViTClassifier
from models.mobilenet import MobileNetV2MRI

class AlzheimerMRIModel:
    def __init__(self, num_classes=None, device=None, weights=None, architecture="mobilenetv2"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if architecture == "vit":
            self.model = ViTClassifier(num_classes=num_classes, weights=weights)
        elif architecture == "mobilenetv2":
            self.model = MobileNetV2MRI(num_classes=num_classes, weights=weights)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def predict(self, x: torch.Tensor, return_probs: bool = False):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            return F.softmax(logits, dim=1) if return_probs else logits

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_model(self):
        return self.model