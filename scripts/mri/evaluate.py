import os
import torch
from torch.utils.data import DataLoader
from domains.mri.dataset import MRIClassifierDataset
from domains.mri.model import AlzheimerMRIModel
from training.evaluator import Evaluator
from common.utils.config import load_config

from torch.utils.data import Subset

if __name__ == "__main__":
    # Absolute path to the root (assuming this file is scripts/mri/evaluate.py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)

    config = load_config("configs/mri/config.yaml")
    device = config["device"]

    eval_dir = os.path.join(project_root, config["paths"]["eval_dir"])

    model_wrapper = AlzheimerMRIModel(
    num_classes=config["model"]["num_classes"],
    device=device,
    weights=None,  # no pretrained weights at inference
    architecture=config["model"]["name"]
    )
    model_path = os.path.join(project_root, config["paths"]["model_dir"], config["paths"]["model_file"])
    model_wrapper.load_weights(model_path)
    model = model_wrapper.get_model()

    train_loader = DataLoader(MRIClassifierDataset(split="train"), batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(MRIClassifierDataset(split="test"), batch_size=config["training"]["batch_size"])

    # Evaluate train
    Evaluator(model, train_loader, device, split_name="train", output_dir=eval_dir).run()

    # test_subset = Subset(MRIClassifierDataset(split="test"), range(50))  # first 50 samples
    # test_loader = DataLoader(test_subset, batch_size=config["training"]["batch_size"])

    # Evaluate test
    Evaluator(model, test_loader, device, split_name="test", output_dir=eval_dir).run()