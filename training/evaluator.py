import torch
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics.classification import compute_classification_metrics

class Evaluator:
    def __init__(self, model, dataloader, device, split_name="test", output_dir=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.split_name = split_name
        self.output_dir = output_dir

    def run(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.dataloader):
                print(f"Evaluating batch {i + 1}/{len(self.dataloader)}")

                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        metrics = compute_classification_metrics(all_labels, all_preds)
        self._log(metrics)
        self._plot_confusion_matrix(metrics["confusion_matrix"])
        return metrics

    def _log(self, metrics):
        print(f"\n[{self.split_name.upper()}] Evaluation Metrics:")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                print(f"{key.capitalize()}: {value:.4f}")

        if self.output_dir is None:
            print("[INFO] output_dir is None - skipping JSON saving.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{self.split_name}_metrics_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)

        print(f"[DEBUG] Saving metrics to: {json_path}")
        try:
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print("Metrics JSON saved successfully!")
        except Exception as e:
            print(f"Failed to save metrics JSON: {e}")

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{self.split_name.capitalize()} Confusion Matrix")
        plt.tight_layout()
        plt.show()