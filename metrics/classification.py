from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np

def compute_classification_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    cm = confusion_matrix(labels, preds)
    num_classes = cm.shape[0]

    # Per-class precision, recall, f1
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0).tolist()
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0).tolist()

    # Per-class specificity
    per_class_specificity = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_specificity.append(specificity)

    macro_specificity = np.mean(per_class_specificity)

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "macro_specificity": macro_specificity,
        "per_class": {
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
            "specificity": per_class_specificity,
        },
        "confusion_matrix": cm.tolist()
    }