from __future__ import annotations

from transformers import AutoModelForTokenClassification


def load_token_classification_model(
    model_path: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
):
    return AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
