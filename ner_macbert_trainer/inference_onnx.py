from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


def strip_table_blocks_from_text(text: str) -> str:
    cleaned = re.sub(r"<table[\s\S]*?</table>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 ONNX 的地质 NER 推理与对齐输出")
    parser.add_argument(
        "--config",
        default="/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/superuser/dev/NER/data",
        help="输入目录，包含上游 .md(JSON数组) 文件",
    )
    parser.add_argument(
        "--onnx-model",
        default="",
        help="ONNX 模型路径，留空时优先使用 INT8，其次 FP32",
    )
    parser.add_argument(
        "--output-bio",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output/infer/predict.bio",
        help="BIO 输出路径",
    )
    parser.add_argument(
        "--output-sentence-map",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output/infer/predict.sentence_map.tsv",
        help="sentence_map 输出路径",
    )
    parser.add_argument(
        "--output-bi-lines",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output/infer/predict.bi_only.tsv",
        help="B/I 行输出路径",
    )
    parser.add_argument(
        "--output-spans",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output/infer/predict.bi_spans.tsv",
        help="实体片段输出路径",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_items_from_directory(data_dir: str) -> list[dict[str, Any]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    items: list[dict[str, Any]] = []
    for file_path in sorted(root.glob("*.md")):
        try:
            records = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(records, list):
            continue
        for block_index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            page_text = record.get("page_text")
            if not isinstance(page_text, str):
                continue
            stripped = strip_table_blocks_from_text(page_text)
            if not stripped:
                continue
            page_number = record.get("page_number")
            items.append(
                {
                    "source_file": file_path.name,
                    "source_path": str(file_path),
                    "page_number": str(page_number) if page_number is not None else "",
                    "block_index": block_index,
                    "text": stripped,
                }
            )
    return items


def build_sentence_id(item: dict[str, Any]) -> str:
    source_file = str(item.get("source_file", "unknown")).replace("\t", " ").replace("\n", " ")
    page_number = str(item.get("page_number", "")).strip() or "NA"
    return f"{source_file}__p{page_number}"


def escape_tsv(value: Any) -> str:
    return str(value).replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def resolve_onnx_path(args_path: str, cfg: dict[str, Any]) -> Path:
    if args_path:
        model_path = Path(args_path)
        if not model_path.exists():
            raise FileNotFoundError(f"指定 ONNX 模型不存在: {model_path}")
        return model_path

    onnx_dir = Path(cfg["output"]["onnx_dir"])
    int8_path = onnx_dir / "ner_macbert_int8.onnx"
    fp32_path = onnx_dir / "ner_macbert_fp32.onnx"
    if int8_path.exists():
        return int8_path
    if fp32_path.exists():
        return fp32_path
    raise FileNotFoundError(f"未找到 ONNX 模型，请先导出: {onnx_dir}")


def load_id2label(saved_model_dir: Path) -> dict[int, str]:
    mapping_path = saved_model_dir / "label_mappings.json"
    if mapping_path.exists():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        id2label_raw = payload.get("id2label", {})
        return {int(k): str(v) for k, v in id2label_raw.items()}

    config_path = saved_model_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    id2label_raw = payload.get("id2label", {})
    return {int(k): str(v) for k, v in id2label_raw.items()}


def predict_tags_for_text(
    text: str,
    tokenizer: AutoTokenizer,
    session: ort.InferenceSession,
    id2label: dict[int, str],
    max_length: int,
    window_stride: int,
) -> list[str]:
    chars = list(text)
    input_names = {item.name for item in session.get_inputs()}
    encoded_batch = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_overflowing_tokens=True,
        stride=max(int(window_stride), 0),
    )

    all_tags: list[str] = ["O"] * len(chars)
    covered: list[bool] = [False] * len(chars)
    window_count = len(encoded_batch["input_ids"])

    for window_idx in range(window_count):
        word_ids = encoded_batch.word_ids(batch_index=window_idx)
        feed = {
            "input_ids": np.asarray([encoded_batch["input_ids"][window_idx]], dtype=np.int64),
            "attention_mask": np.asarray([encoded_batch["attention_mask"][window_idx]], dtype=np.int64),
        }
        if "token_type_ids" in encoded_batch and "token_type_ids" in input_names:
            feed["token_type_ids"] = np.asarray([encoded_batch["token_type_ids"][window_idx]], dtype=np.int64)

        logits = session.run(["logits"], feed)[0]
        pred_ids = np.argmax(logits[0], axis=-1).tolist()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(chars):
                continue
            predicted_tag = id2label.get(int(pred_ids[token_idx]), "O")
            if not covered[word_idx]:
                all_tags[word_idx] = predicted_tag
                covered[word_idx] = True
            elif all_tags[word_idx] == "O" and predicted_tag != "O":
                all_tags[word_idx] = predicted_tag

    return all_tags


def tags_to_spans(tags: list[str], text: str) -> list[tuple[str, int, int, str]]:
    spans: list[tuple[str, int, int, str]] = []
    current_type = ""
    start = -1
    chars: list[str] = []

    def flush(end_idx: int) -> None:
        nonlocal current_type, start, chars
        if current_type and start >= 0 and chars:
            spans.append((current_type, start, end_idx, "".join(chars)))
        current_type = ""
        start = -1
        chars = []

    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            flush(idx - 1)
            current_type = tag[2:]
            start = idx
            chars = [text[idx]]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_type == entity_type and start >= 0:
                chars.append(text[idx])
            else:
                flush(idx - 1)
                current_type = entity_type
                start = idx
                chars = [text[idx]]
        else:
            flush(idx - 1)
    flush(len(tags) - 1)
    return spans


def normalize_bio_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    prev_type = ""
    prev_prefix = "O"
    for tag in tags:
        if tag.startswith("B-"):
            entity_type = tag[2:]
            normalized.append(tag)
            prev_type = entity_type
            prev_prefix = "B"
            continue
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if prev_prefix in {"B", "I"} and prev_type == entity_type:
                normalized.append(tag)
                prev_type = entity_type
                prev_prefix = "I"
            else:
                normalized.append(f"B-{entity_type}")
                prev_type = entity_type
                prev_prefix = "B"
            continue
        normalized.append("O")
        prev_type = ""
        prev_prefix = "O"
    return normalized


def apply_span_constraints(tags: list[str], text: str, constraints: dict[str, dict[str, int]]) -> list[str]:
    if not constraints:
        return tags
    adjusted = list(tags)
    for entity_type, start, end, span_text in tags_to_spans(tags, text):
        rule = constraints.get(entity_type)
        if not rule:
            continue
        length = len(span_text)
        min_len = int(rule.get("min_len", 1))
        max_len = int(rule.get("max_len", 10**9))
        if length < min_len or length > max_len:
            for idx in range(start, end + 1):
                adjusted[idx] = "O"
    return normalize_bio_tags(adjusted)


def apply_texture_lexicon(tags: list[str], text: str, texture_lexicon: list[str]) -> list[str]:
    if not texture_lexicon:
        return tags
    adjusted = list(tags)
    candidates = sorted({item.strip() for item in texture_lexicon if item.strip()}, key=len, reverse=True)
    for term in candidates:
        if len(term) < 2:
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            end = idx + len(term) - 1
            ctx_left = max(0, idx - 6)
            ctx_right = min(len(text), end + 7)
            context = text[ctx_left:ctx_right]
            if all(adjusted[pos] == "O" for pos in range(idx, end + 1)) and any(
                marker in context for marker in ["岩", "砂", "泥", "砾", "页", "灰"]
            ):
                adjusted[idx] = "B-lithology_texture"
                for pos in range(idx + 1, end + 1):
                    adjusted[pos] = "I-lithology_texture"
            start = idx + 1
    return normalize_bio_tags(adjusted)


def write_outputs(
    items: list[dict[str, Any]],
    predictions: list[list[str]],
    output_bio: Path,
    output_sentence_map: Path,
    output_bi_lines: Path,
    output_spans: Path,
    span_constraints: dict[str, dict[str, int]],
    texture_lexicon: list[str],
) -> None:
    bio_lines: list[str] = []
    sentence_map_rows: list[str] = [
        "sentence_index\tsentence_id\tsource_file\tsource_path\tpage_number\tblock_index\ttext_length\ttext"
    ]
    bi_rows: list[str] = ["sentence_id\ttoken_idx\tchar\ttag"]
    span_rows: list[str] = ["sentence_id\tentity_type\tstart_idx\tend_idx\ttext"]

    for idx, (item, tags) in enumerate(zip(items, predictions, strict=True)):
        text = str(item["text"])
        tags = normalize_bio_tags(tags)
        tags = apply_span_constraints(tags, text, span_constraints)
        tags = apply_texture_lexicon(tags, text, texture_lexicon)
        sentence_id = build_sentence_id(item)

        for ch, tag in zip(text, tags, strict=True):
            bio_lines.append(f"{ch} {tag}")
        bio_lines.append("")

        sentence_map_rows.append(
            "\t".join(
                [
                    str(idx),
                    escape_tsv(sentence_id),
                    escape_tsv(item["source_file"]),
                    escape_tsv(item["source_path"]),
                    escape_tsv(item["page_number"]),
                    escape_tsv(item["block_index"]),
                    str(len(text)),
                    escape_tsv(text),
                ]
            )
        )

        for token_idx, (ch, tag) in enumerate(zip(text, tags, strict=True)):
            if tag.startswith(("B-", "I-")):
                bi_rows.append(f"{sentence_id}\t{token_idx}\t{ch}\t{tag}")

        for entity_type, start, end, span_text in tags_to_spans(tags, text):
            span_rows.append(f"{sentence_id}\t{entity_type}\t{start}\t{end}\t{span_text}")

    output_bio.parent.mkdir(parents=True, exist_ok=True)
    output_sentence_map.parent.mkdir(parents=True, exist_ok=True)
    output_bi_lines.parent.mkdir(parents=True, exist_ok=True)
    output_spans.parent.mkdir(parents=True, exist_ok=True)
    output_bio.write_text("\n".join(bio_lines), encoding="utf-8")
    output_sentence_map.write_text("\n".join(sentence_map_rows) + "\n", encoding="utf-8")
    output_bi_lines.write_text("\n".join(bi_rows) + "\n", encoding="utf-8")
    output_spans.write_text("\n".join(span_rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    saved_model_dir = Path(cfg["output"]["saved_model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
    id2label = load_id2label(saved_model_dir)
    max_length = int(cfg["runtime"]["max_length"])
    infer_window_stride = int(cfg["runtime"].get("infer_window_stride", 64))
    span_constraints = cfg["runtime"].get("span_constraints", {})
    texture_lexicon = cfg["runtime"].get("texture_lexicon", [])
    onnx_path = resolve_onnx_path(args.onnx_model, cfg)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    items = load_items_from_directory(args.input_dir)
    predictions: list[list[str]] = []
    progress = tqdm(items, total=len(items), desc="推理进度", unit="page_text", dynamic_ncols=True)
    for item in progress:
        progress.set_postfix(
            file=item.get("source_file", ""),
            page=item.get("page_number", ""),
            refresh=True,
        )
        predictions.append(
            predict_tags_for_text(
                text=str(item["text"]),
                tokenizer=tokenizer,
                session=session,
                id2label=id2label,
                max_length=max_length,
                window_stride=infer_window_stride,
            )
        )
    progress.close()

    write_outputs(
        items=items,
        predictions=predictions,
        output_bio=Path(args.output_bio),
        output_sentence_map=Path(args.output_sentence_map),
        output_bi_lines=Path(args.output_bi_lines),
        output_spans=Path(args.output_spans),
        span_constraints=span_constraints,
        texture_lexicon=texture_lexicon,
    )
    print(f"推理完成，共处理 {len(items)} 条 page_text")
    print(f"BIO 输出: {args.output_bio}")
    print(f"sentence_map 输出: {args.output_sentence_map}")
    print(f"B/I 行输出: {args.output_bi_lines}")
    print(f"实体片段输出: {args.output_spans}")


if __name__ == "__main__":
    main()
