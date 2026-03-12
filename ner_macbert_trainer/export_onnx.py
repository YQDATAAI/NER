from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 NER 模型到 ONNX 并执行 INT8 动态量化")
    parser.add_argument(
        "--config",
        default="/home/superuser/dev/NER/ner_macbert_trainer/conf/training_args.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    saved_model_dir = Path(cfg["output"]["saved_model_dir"])
    onnx_dir = Path(cfg["output"]["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=4, desc="导出进度", unit="step")

    model = AutoModelForTokenClassification.from_pretrained(saved_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
    model.eval()
    progress.update(1)
    progress.set_postfix(stage="模型与Tokenizer加载")

    sample_inputs = tokenizer("地质样本", return_tensors="pt")
    input_names = [name for name in ["input_ids", "attention_mask", "token_type_ids"] if name in sample_inputs]
    input_tensors = tuple(sample_inputs[name] for name in input_names)

    fp32_path = onnx_dir / "ner_macbert_fp32.onnx"
    dynamic_axes = {name: {0: "batch_size", 1: "sequence_length"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensors,
            fp32_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            do_constant_folding=True,
        )
    progress.update(1)
    progress.set_postfix(stage="FP32 ONNX导出")

    int8_path = onnx_dir / "ner_macbert_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    progress.update(1)
    progress.set_postfix(stage="INT8量化")

    progress.update(1)
    progress.set_postfix(stage="导出完成")
    progress.close()

    print(f"FP32 ONNX 已导出: {fp32_path}")
    print(f"INT8 ONNX 已导出: {int8_path}")


if __name__ == "__main__":
    main()
