from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def parse_bio_file(bio_path: str) -> tuple[list[list[str]], list[list[str]]]:
    path = Path(bio_path)
    if not path.exists():
        raise FileNotFoundError(f"BIO 文件不存在: {bio_path}")

    sentences: list[list[str]] = []
    tags: list[list[str]] = []
    current_tokens: list[str] = []
    current_tags: list[str] = []

    pattern = re.compile(r"^(.*?)[ \t]+(\S+)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line == "":
            if current_tokens:
                sentences.append(current_tokens)
                tags.append(current_tags)
                current_tokens = []
                current_tags = []
            continue

        if "\t" in raw_line:
            parts = raw_line.rsplit("\t", maxsplit=1)
            token = parts[0]
            tag = parts[1].strip()
        else:
            match = pattern.match(raw_line)
            if match is None:
                raise ValueError(f"无法解析 BIO 行: {raw_line}")
            token, tag = match.group(1), match.group(2)

        if tag == "":
            raise ValueError(f"无法解析 BIO 行: {raw_line}")
        current_tokens.append(token)
        current_tags.append(tag)

    if current_tokens:
        sentences.append(current_tokens)
        tags.append(current_tags)

    if not sentences:
        raise ValueError(f"BIO 文件为空: {bio_path}")

    return sentences, tags


def strip_table_blocks(
    sentences: list[list[str]],
    tags: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], list[int]]:
    cleaned_sentences: list[list[str]] = []
    cleaned_tags: list[list[str]] = []
    kept_indices: list[int] = []
    pattern = re.compile(r"<table[\s\S]*?</table>", flags=re.IGNORECASE)
    for idx, (sentence_tokens, sentence_tags) in enumerate(zip(sentences, tags, strict=True)):
        text = "".join(sentence_tokens)
        mask = [True] * len(sentence_tokens)
        has_table = False
        for match in pattern.finditer(text):
            has_table = True
            start, end = match.span()
            for pos in range(start, end):
                if 0 <= pos < len(mask):
                    mask[pos] = False
        if not has_table:
            cleaned_sentences.append(sentence_tokens)
            cleaned_tags.append(sentence_tags)
            kept_indices.append(idx)
            continue
        filtered_tokens = [tok for tok, keep in zip(sentence_tokens, mask, strict=True) if keep]
        filtered_tags = [tag for tag, keep in zip(sentence_tags, mask, strict=True) if keep]
        if filtered_tokens:
            cleaned_sentences.append(filtered_tokens)
            cleaned_tags.append(filtered_tags)
            kept_indices.append(idx)
    return cleaned_sentences, cleaned_tags, kept_indices


def build_label_mappings(all_tags: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted({tag for sentence_tags in all_tags for tag in sentence_tags})
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _subtoken_label(tag: str) -> str:
    if tag.startswith("B-"):
        return f"I-{tag[2:]}"
    return tag


def tokenize_and_align_labels(
    tokens: list[str],
    tags: list[str],
    tokenizer: PreTrainedTokenizerBase,
    label2id: dict[str, int],
    max_length: int,
) -> tuple[dict[str, list[int]], list[int]]:
    windows = tokenize_and_align_labels_windows(
        tokens=tokens,
        tags=tags,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        window_stride=0,
    )
    return windows[0]


def tokenize_and_align_labels_windows(
    tokens: list[str],
    tags: list[str],
    tokenizer: PreTrainedTokenizerBase,
    label2id: dict[str, int],
    max_length: int,
    window_stride: int,
) -> list[tuple[dict[str, list[int]], list[int]]]:
    encoded_batch = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=max(int(window_stride), 0),
        return_attention_mask=True,
    )

    window_count = len(encoded_batch["input_ids"])
    windows: list[tuple[dict[str, list[int]], list[int]]] = []
    for window_idx in range(window_count):
        word_ids = encoded_batch.word_ids(batch_index=window_idx)
        aligned_labels: list[int] = []
        previous_word_id: int | None = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
                continue
            source_tag = tags[word_id]
            if word_id != previous_word_id:
                aligned_labels.append(label2id[source_tag])
            else:
                aligned_labels.append(label2id.get(_subtoken_label(source_tag), label2id[source_tag]))
            previous_word_id = word_id

        encoded_window: dict[str, list[int]] = {
            "input_ids": list(encoded_batch["input_ids"][window_idx]),
            "attention_mask": list(encoded_batch["attention_mask"][window_idx]),
        }
        if "token_type_ids" in encoded_batch:
            encoded_window["token_type_ids"] = list(encoded_batch["token_type_ids"][window_idx])

        if len(aligned_labels) != len(encoded_window["input_ids"]):
            raise ValueError("标签对齐失败：labels 与 input_ids 长度不一致")
        windows.append((encoded_window, aligned_labels))
    return windows


@dataclass
class EncodedSample:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    labels: list[int]


class NERDataset(Dataset):
    def __init__(
        self,
        sentences: list[list[str]],
        tags: list[list[str]],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int,
        window_stride: int = 0,
        positive_upsample: int = 1,
        o_label_id: int | None = None,
        texture_upsample: int = 1,
        texture_label_ids: set[int] | None = None,
    ) -> None:
        self.samples: list[EncodedSample] = []
        for sentence_tokens, sentence_tags in zip(sentences, tags, strict=True):
            windows = tokenize_and_align_labels_windows(
                tokens=sentence_tokens,
                tags=sentence_tags,
                tokenizer=tokenizer,
                label2id=label2id,
                max_length=max_length,
                window_stride=window_stride,
            )
            for encoded, labels in windows:
                sample = EncodedSample(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                    labels=labels,
                )
                self.samples.append(sample)
                if positive_upsample > 1 and o_label_id is not None and any(
                    label >= 0 and label != o_label_id for label in labels
                ):
                    for _ in range(int(positive_upsample) - 1):
                        self.samples.append(
                            EncodedSample(
                                input_ids=list(sample.input_ids),
                                attention_mask=list(sample.attention_mask),
                                token_type_ids=list(sample.token_type_ids) if sample.token_type_ids is not None else None,
                                labels=list(sample.labels),
                            )
                        )
                if texture_upsample > 1 and texture_label_ids and any(
                    label in texture_label_ids for label in labels
                ):
                    for _ in range(int(texture_upsample) - 1):
                        self.samples.append(
                            EncodedSample(
                                input_ids=list(sample.input_ids),
                                attention_mask=list(sample.attention_mask),
                                token_type_ids=list(sample.token_type_ids) if sample.token_type_ids is not None else None,
                                labels=list(sample.labels),
                            )
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        sample = self.samples[idx]
        item = {
            "input_ids": sample.input_ids,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
        }
        if sample.token_type_ids is not None:
            item["token_type_ids"] = sample.token_type_ids
        return item
