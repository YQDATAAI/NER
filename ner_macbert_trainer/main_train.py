from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from core import NERDataset, build_compute_metrics, build_label_mappings, load_token_classification_model, parse_bio_file, strip_table_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MacBERT-large 地质 NER 分布式训练入口")
    parser.add_argument(
        "--config",
        default="/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.yaml",
        help="训练配置文件路径",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    sentences: list[list[str]],
    tags: list[list[str]],
    seed: int,
) -> tuple[tuple[list[list[str]], list[list[str]]], tuple[list[list[str]], list[list[str]]], tuple[list[list[str]], list[list[str]]]]:
    indices = list(range(len(sentences)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * 0.8)
    n_eval = int(n_total * 0.1)
    n_test = n_total - n_train - n_eval

    if n_train == 0:
        n_train = 1
    if n_eval == 0 and n_total >= 3:
        n_eval = 1
    if n_test == 0 and n_total >= 2:
        n_test = 1
    while n_train + n_eval + n_test > n_total:
        if n_train >= n_eval and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_eval >= n_test and n_eval > 0:
            n_eval -= 1
        else:
            n_test -= 1

    train_idx = indices[:n_train]
    eval_idx = indices[n_train : n_train + n_eval]
    test_idx = indices[n_train + n_eval : n_train + n_eval + n_test]

    def _select(sel: list[int]) -> tuple[list[list[str]], list[list[str]]]:
        return [sentences[i] for i in sel], [tags[i] for i in sel]

    return _select(train_idx), _select(eval_idx), _select(test_idx)


def load_group_keys_from_sentence_map(
    bio_path: str,
    sentence_lengths: list[int],
) -> list[str] | None:
    sentence_map_path = Path(bio_path).with_suffix(".sentence_map.tsv")
    if not sentence_map_path.exists():
        return None
    chunk_lengths: list[int] = []
    chunk_groups: list[str] = []
    with sentence_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                length = int(row.get("text_length", "0"))
            except Exception:
                continue
            if length <= 0:
                continue
            source_file = str(row.get("source_file", "")).strip()
            chunk_lengths.append(length)
            chunk_groups.append(source_file if source_file else "unknown")
    if not chunk_lengths:
        return None

    chunk_ends: list[int] = []
    total = 0
    for length in chunk_lengths:
        total += length
        chunk_ends.append(total)

    group_keys: list[str] = []
    offset = 0
    chunk_idx = 0
    for length in sentence_lengths:
        while chunk_idx < len(chunk_ends) and offset >= chunk_ends[chunk_idx]:
            chunk_idx += 1
        if chunk_idx >= len(chunk_groups):
            group_keys.append(chunk_groups[-1])
        else:
            group_keys.append(chunk_groups[chunk_idx])
        offset += length
    return group_keys


def split_dataset_by_group(
    sentences: list[list[str]],
    tags: list[list[str]],
    group_keys: list[str],
    seed: int,
) -> tuple[tuple[list[list[str]], list[list[str]]], tuple[list[list[str]], list[list[str]]], tuple[list[list[str]], list[list[str]]]]:
    if len(group_keys) != len(sentences):
        return split_dataset(sentences, tags, seed=seed)

    group_to_indices: dict[str, list[int]] = {}
    for idx, key in enumerate(group_keys):
        group_to_indices.setdefault(key, []).append(idx)
    groups = list(group_to_indices.items())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    n_total = len(sentences)
    target_train = max(int(n_total * 0.8), 1)
    target_eval = max(int(n_total * 0.1), 1)
    target_test = n_total - target_train - target_eval
    if target_test <= 0:
        target_test = 1
        if target_train > target_eval:
            target_train -= 1
        else:
            target_eval -= 1

    train_idx: list[int] = []
    eval_idx: list[int] = []
    test_idx: list[int] = []
    for _, indices in groups:
        deficits = {
            "train": target_train - len(train_idx),
            "eval": target_eval - len(eval_idx),
            "test": target_test - len(test_idx),
        }
        target_split = max(deficits.items(), key=lambda x: x[1])[0]
        if target_split == "train":
            train_idx.extend(indices)
        elif target_split == "eval":
            eval_idx.extend(indices)
        else:
            test_idx.extend(indices)

    if not eval_idx and len(train_idx) > 1:
        eval_idx.append(train_idx.pop())
    if not test_idx and len(train_idx) > 1:
        test_idx.append(train_idx.pop())

    def _select(sel: list[int]) -> tuple[list[list[str]], list[list[str]]]:
        return [sentences[i] for i in sel], [tags[i] for i in sel]

    return _select(train_idx), _select(eval_idx), _select(test_idx)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    output_cfg = cfg["output"]
    Path(output_cfg["root_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_cfg["saved_model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_cfg["onnx_dir"]).mkdir(parents=True, exist_ok=True)


def build_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    train_cfg = cfg["train"]
    training_kwargs: dict[str, Any] = {
        "output_dir": cfg["output"]["checkpoint_dir"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg["per_device_eval_batch_size"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "save_total_limit": int(train_cfg["save_total_limit"]),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "fp16": bool(cfg["runtime"]["fp16"]),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "report_to": "none",
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    if "disable_tqdm" in params:
        training_kwargs["disable_tqdm"] = False
    if "logging_strategy" in params:
        training_kwargs["logging_strategy"] = "steps"
    if "logging_first_step" in params:
        training_kwargs["logging_first_step"] = False
    if "do_train" in params:
        training_kwargs["do_train"] = True
    if "do_eval" in params:
        training_kwargs["do_eval"] = True

    return TrainingArguments(**training_kwargs)


def build_label_weights(label2id: dict[str, int], cfg: dict[str, Any]) -> torch.Tensor | None:
    runtime_cfg = cfg.get("runtime", {})
    texture_boost = float(runtime_cfg.get("texture_loss_boost", 1.0))
    if texture_boost <= 1.0:
        return None
    weights = torch.ones(len(label2id), dtype=torch.float32)
    for label, idx in label2id.items():
        if label.endswith("lithology_texture"):
            weights[idx] = texture_boost
    return weights


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, label_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.label_weights.to(logits.device), ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss


def build_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset: NERDataset,
    eval_dataset: NERDataset,
    tokenizer,
    data_collator,
    compute_metrics,
    label_weights: torch.Tensor | None,
) -> Trainer:
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    if label_weights is not None:
        return WeightedLossTrainer(label_weights=label_weights, **trainer_kwargs)
    return Trainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_global_seed(int(cfg["runtime"]["seed"]))

    sentences, tags = parse_bio_file(cfg["data"]["train_bio_path"])
    sentence_lengths = [len(x) for x in sentences]
    group_keys = load_group_keys_from_sentence_map(cfg["data"]["train_bio_path"], sentence_lengths)
    if bool(cfg["runtime"].get("strip_table_blocks", True)):
        sentences, tags, kept_indices = strip_table_blocks(sentences, tags)
        if group_keys is not None:
            group_keys = [group_keys[i] for i in kept_indices]

    label2id, id2label = build_label_mappings(tags)
    if bool(cfg["runtime"].get("split_by_source_file", True)) and group_keys is not None:
        (train_sentences, train_tags), (eval_sentences, eval_tags), (test_sentences, test_tags) = split_dataset_by_group(
            sentences,
            tags,
            group_keys=group_keys,
            seed=int(cfg["runtime"]["seed"]),
        )
    else:
        (train_sentences, train_tags), (eval_sentences, eval_tags), (test_sentences, test_tags) = split_dataset(
            sentences,
            tags,
            seed=int(cfg["runtime"]["seed"]),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_path"], use_fast=True)
    max_length = int(cfg["runtime"]["max_length"])
    train_window_stride = int(cfg["runtime"].get("train_window_stride", 64))
    positive_upsample = int(cfg["runtime"].get("train_positive_upsample", 1))
    texture_upsample = int(cfg["runtime"].get("train_texture_upsample", 1))
    o_label_id = label2id.get("O")
    texture_label_ids = {idx for label, idx in label2id.items() if label.endswith("lithology_texture")}

    train_dataset = NERDataset(
        train_sentences,
        train_tags,
        tokenizer,
        label2id,
        max_length=max_length,
        window_stride=train_window_stride,
        positive_upsample=positive_upsample,
        o_label_id=o_label_id,
        texture_upsample=texture_upsample,
        texture_label_ids=texture_label_ids,
    )
    eval_dataset = NERDataset(
        eval_sentences,
        eval_tags,
        tokenizer,
        label2id,
        max_length=max_length,
        window_stride=train_window_stride,
    )
    test_dataset = NERDataset(
        test_sentences,
        test_tags,
        tokenizer,
        label2id,
        max_length=max_length,
        window_stride=train_window_stride,
    )

    model = load_token_classification_model(
        model_path=cfg["model"]["pretrained_model_path"],
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics(id2label=id2label)
    label_weights = build_label_weights(label2id, cfg)

    training_args = build_training_args(cfg)

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        label_weights=label_weights,
    )

    trainer.train()
    test_metrics = trainer.predict(test_dataset).metrics

    saved_model_dir = Path(cfg["output"]["saved_model_dir"])
    trainer.save_model(str(saved_model_dir))
    tokenizer.save_pretrained(str(saved_model_dir))
    with (saved_model_dir / "label_mappings.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, ensure_ascii=False, indent=2)

    metrics_path = Path(cfg["output"]["root_dir"]) / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"训练完成，模型已保存至: {saved_model_dir}")
    print(f"Test 集指标: {json.dumps(test_metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
