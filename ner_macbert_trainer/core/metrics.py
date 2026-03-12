from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


def build_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for prediction, label in zip(predictions, labels, strict=True):
            pred_tags: list[str] = []
            label_tags: list[str] = []
            for pred_id, label_id in zip(prediction, label, strict=True):
                if label_id == -100:
                    continue
                pred_tags.append(id2label[int(pred_id)])
                label_tags.append(id2label[int(label_id)])
            true_predictions.append(pred_tags)
            true_labels.append(label_tags)

        result = {
            "precision": float(precision_score(true_labels, true_predictions)),
            "recall": float(recall_score(true_labels, true_predictions)),
            "f1": float(f1_score(true_labels, true_predictions)),
        }

        report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
        for name, values in report.items():
            if name in {"micro avg", "macro avg", "weighted avg"}:
                continue
            if isinstance(values, dict) and name != "O":
                result[f"f1_{name}"] = float(values.get("f1-score", 0.0))
        return result

    return compute_metrics
