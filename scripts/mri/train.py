import os
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import ViT_B_16_Weights, MobileNet_V2_Weights
from domains.mri.dataset import MRIClassifierDataset
from domains.mri.model import AlzheimerMRIModel
from training.trainer import Trainer
from common.utils.config import load_config
from common.utils.seed import set_seed

if __name__ == "__main__":
    # Absolute path to the root (assuming this file is scripts/mri/train.py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(project_root)

    # Auto-select weights based on architecture
    config = load_config("configs/mri/config.yaml")
    architecture = config["model"]["name"]
    if architecture == "vit":
        weights = ViT_B_16_Weights.DEFAULT
    elif architecture == "mobilenetv2":
        weights = MobileNet_V2_Weights.DEFAULT
    else:
        weights = None  # in case you want to train from scratch

    set_seed(config["training"]["seed"])
    device = config["device"]

    output_path = config["paths"]["model_dir"]
    log_dir = config["paths"].get("log_dir")
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Setup
    model_wrapper = AlzheimerMRIModel(
        num_classes=config["model"]["num_classes"],
        device=device,
        weights=weights,
        architecture=architecture
    )
    model = model_wrapper.get_model()

    train_dataset = MRIClassifierDataset(split=config["dataset"]["split"])
    train_dataset = Subset(train_dataset, range(20))  # only 20 samples
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    trainer = Trainer(model, train_loader, criterion, optimizer, device, log_dir=log_dir)
    trainer.train(epochs=config["training"]["epochs"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{architecture}_mri_{timestamp}.pth"
    model_path = os.path.join(output_path, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")