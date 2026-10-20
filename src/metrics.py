from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    if y_true.size == 0:
        return 0.0

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    contingency = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        contingency[pred_label, true_label] += 1

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / y_true.size)


@torch.no_grad()
def evaluate_clustering(model, dataloader, device: torch.device) -> Dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        batch_labels = batch["label"].cpu().numpy()

        teacher_features = model.encode_teacher(images)
        student_features = model.encode_student(teacher_features=teacher_features)
        logits = model.prototype_logits(student_features)
        preds = logits.argmax(dim=1).cpu().numpy()

        predictions.append(preds)
        labels.append(batch_labels)

    predictions_np = np.concatenate(predictions)
    labels_np = np.concatenate(labels)
    return {
        "acc": clustering_accuracy(labels_np, predictions_np),
        "nmi": float(normalized_mutual_info_score(labels_np, predictions_np)),
        "ari": float(adjusted_rand_score(labels_np, predictions_np)),
    }


def clustering_selection_score(metrics: Mapping[str, float]) -> float:
    return float(metrics["acc"] + metrics["nmi"] + metrics["ari"])


def format_clustering_metrics(metrics: Mapping[str, float]) -> str:
    return "acc={acc:.4f} nmi={nmi:.4f} ari={ari:.4f}".format(
        acc=metrics["acc"],
        nmi=metrics["nmi"],
        ari=metrics["ari"],
    )
