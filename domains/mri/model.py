import torch
from torch import nn
import torch.nn.functional as F
from models.vit import ViTClassifier

class AlzheimerMRIModel:
    def __init__(self, num_classes=4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTClassifier(num_classes=num_classes).to(self.device)

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