import torch
from models.vit import ViTClassifier
from domains.mri.model import AlzheimerMRIModel

def test_vit_output_shape():
    model = ViTClassifier(num_classes=4)
    x = torch.randn(2, 3, 224, 224)  # batch of 2 images
    out = model(x)
    assert out.shape == (2, 4)

def test_predict_output_shape():
    model = AlzheimerMRIModel(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model.predict(x)
    assert out.shape == (2, 4)