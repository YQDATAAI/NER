from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer

from inference_onnx import (
    apply_span_constraints,
    apply_texture_lexicon,
    load_config,
    load_id2label,
    load_items_from_directory,
    normalize_bio_tags,
    predict_tags_for_text,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双ONNX模型融合推理：总体优先 + 纹理优先")
    parser.add_argument(
        "--base-config",
        default="/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.dual_base.yaml",
        help="总体优先模型配置",
    )
    parser.add_argument(
        "--texture-config",
        default="/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.dual_texture.yaml",
        help="纹理优先模型配置",
    )
    parser.add_argument(
        "--input-dir",
        default="/home/superuser/dev/NER/data",
        help="输入目录",
    )
    parser.add_argument(
        "--base-onnx",
        default="",
        help="总体优先 ONNX 路径，留空自动从base配置读取",
    )
    parser.add_argument(
        "--texture-onnx",
        default="",
        help="纹理优先 ONNX 路径，留空自动从texture配置读取",
    )
    parser.add_argument(
        "--output-bio",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output_dual/fused/predict.bio",
        help="BIO 输出路径",
    )
    parser.add_argument(
        "--output-sentence-map",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output_dual/fused/predict.sentence_map.tsv",
        help="sentence_map 输出路径",
    )
    parser.add_argument(
        "--output-bi-lines",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output_dual/fused/predict.bi_only.tsv",
        help="B/I 行输出路径",
    )
    parser.add_argument(
        "--output-spans",
        default="/home/superuser/dev/NER/ner_macbert_trainer/output_dual/fused/predict.bi_spans.tsv",
        help="实体片段输出路径",
    )
    return parser.parse_args()


def resolve_onnx(config: dict, explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX 不存在: {path}")
        return path
    onnx_dir = Path(config["output"]["onnx_dir"])
    int8_path = onnx_dir / "ner_macbert_int8.onnx"
    fp32_path = onnx_dir / "ner_macbert_fp32.onnx"
    if int8_path.exists():
        return int8_path
    if fp32_path.exists():
        return fp32_path
    raise FileNotFoundError(f"未找到ONNX: {onnx_dir}")


def fuse_tags(base_tags: list[str], texture_tags: list[str]) -> list[str]:
    if len(base_tags) != len(texture_tags):
        raise ValueError("融合失败：标签长度不一致")
    fused = list(base_tags)
    for idx, (base_tag, texture_tag) in enumerate(zip(base_tags, texture_tags, strict=True)):
        if texture_tag.startswith("B-lithology_texture") or texture_tag.startswith("I-lithology_texture"):
            if base_tag == "O" or base_tag.endswith("lithology_texture"):
                fused[idx] = texture_tag
    return normalize_bio_tags(fused)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    texture_cfg = load_config(args.texture_config)
    items = load_items_from_directory(args.input_dir)

    base_saved_dir = Path(base_cfg["output"]["saved_model_dir"])
    texture_saved_dir = Path(texture_cfg["output"]["saved_model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(base_saved_dir, use_fast=True)
    base_id2label = load_id2label(base_saved_dir)
    texture_id2label = load_id2label(texture_saved_dir)

    base_session = ort.InferenceSession(str(resolve_onnx(base_cfg, args.base_onnx)), providers=["CPUExecutionProvider"])
    texture_session = ort.InferenceSession(
        str(resolve_onnx(texture_cfg, args.texture_onnx)), providers=["CPUExecutionProvider"]
    )
    max_length = int(base_cfg["runtime"]["max_length"])
    infer_window_stride = int(base_cfg["runtime"].get("infer_window_stride", 64))

    predictions: list[list[str]] = []
    progress = tqdm(items, total=len(items), desc="融合推理进度", unit="page_text", dynamic_ncols=True)
    for item in progress:
        text = str(item["text"])
        base_tags = predict_tags_for_text(
            text=text,
            tokenizer=tokenizer,
            session=base_session,
            id2label=base_id2label,
            max_length=max_length,
            window_stride=infer_window_stride,
        )
        texture_tags = predict_tags_for_text(
            text=text,
            tokenizer=tokenizer,
            session=texture_session,
            id2label=texture_id2label,
            max_length=max_length,
            window_stride=infer_window_stride,
        )
        merged = fuse_tags(base_tags, texture_tags)
        merged = apply_span_constraints(merged, text, base_cfg["runtime"].get("span_constraints", {}))
        merged = apply_texture_lexicon(merged, text, base_cfg["runtime"].get("texture_lexicon", []))
        predictions.append(merged)
    progress.close()

    write_outputs(
        items=items,
        predictions=predictions,
        output_bio=Path(args.output_bio),
        output_sentence_map=Path(args.output_sentence_map),
        output_bi_lines=Path(args.output_bi_lines),
        output_spans=Path(args.output_spans),
        span_constraints=base_cfg["runtime"].get("span_constraints", {}),
        texture_lexicon=[],
    )

    print(f"融合推理完成，共处理 {len(items)} 条 page_text")
    print(f"BIO 输出: {args.output_bio}")
    print(f"sentence_map 输出: {args.output_sentence_map}")
    print(f"B/I 行输出: {args.output_bi_lines}")
    print(f"实体片段输出: {args.output_spans}")


if __name__ == "__main__":
    main()
