from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np

def compute_classification_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)  # Sensitivity
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    cm = confusion_matrix(labels, preds)

    # Compute specificity per class and take macro average
    specificity_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,          # Sensitivity
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": cm.tolist()
    }