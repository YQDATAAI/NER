import argparse
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Any

import torch
from core import BIOConverter, DataLoader, LLMOutputParser, PromptBuilder, QwenModelEngine
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Qwen3.5-4B 的 NER 数据集自动构建与 BIO 转换")
    parser.add_argument("--input-dir", default="/home/superuser/dev/NER/data", help="输入目录，包含 .md(JSON数组) 文件")
    parser.add_argument(
        "--model-path",
        default="/home/superuser/LLM_Model/Qwen3.5-4B",
        help="本地模型路径",
    )
    parser.add_argument(
        "--prompt-config",
        default="/home/superuser/dev/NER/ner_dataset_builder/configs/prompt_config.json",
        help="Prompt 配置文件路径",
    )
    parser.add_argument(
        "--few-shots",
        default="/home/superuser/dev/NER/ner_dataset_builder/configs/few_shots.yaml",
        help="Few-shots YAML 路径",
    )
    parser.add_argument(
        "--output-file",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.bio",
        help="BIO 格式输出文件",
    )
    parser.add_argument(
        "--sentence-map-file",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.sentence_map.tsv",
        help="sentence_id 与来源映射文件路径",
    )
    parser.add_argument(
        "--rules-config",
        default="/home/superuser/dev/NER/ner_dataset_builder/configs/entity_rules.yaml",
        help="实体规则配置路径，可为空文件或不存在",
    )
    parser.add_argument(
        "--rules-audit-file",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/rules.audit.tsv",
        help="规则过滤审计输出路径，可为空文件或不存在",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="并行 worker 数")
    parser.add_argument("--gpu-ids", default="", help="并行使用的 GPU 编号，逗号分隔")
    parser.add_argument("--max-input-chars", type=int, default=0, help="输入最大字符数，0 表示不裁剪")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="覆盖生成最大 token，0 表示使用配置默认值")
    return parser.parse_args()


def build_sentence_id(item: dict[str, object]) -> str:
    source_file = str(item.get("source_file", "unknown")).replace("\t", " ").replace("\n", " ")
    page_number = str(item.get("page_number", "")).strip() or "NA"
    return f"{source_file}__p{page_number}"


def escape_tsv(value: object) -> str:
    return str(value).replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def preprocess_text(text: str, max_input_chars: int) -> str:
    if max_input_chars <= 0 or len(text) <= max_input_chars:
        return text
    return text[:max_input_chars]


def build_sentence_map_row(sentence_index: int, sentence_id: str, item: dict[str, Any], text: str) -> str:
    return "\t".join(
        [
            str(sentence_index),
            escape_tsv(sentence_id),
            escape_tsv(item["source_file"]),
            escape_tsv(item["source_path"]),
            escape_tsv(item["page_number"]),
            escape_tsv(item["block_index"]),
            str(len(text)),
            escape_tsv(text),
        ]
    )


def build_rules_audit_rows(sentence_id: str, item: dict[str, Any], filtered_entities: list[dict[str, str]]) -> list[str]:
    rows: list[str] = []
    for filtered in filtered_entities:
        rows.append(
            "\t".join(
                [
                    escape_tsv(sentence_id),
                    escape_tsv(item["source_file"]),
                    escape_tsv(item["page_number"]),
                    escape_tsv(item["block_index"]),
                    escape_tsv(filtered.get("entity_type", "")),
                    escape_tsv(filtered.get("entity_text", "")),
                    escape_tsv(filtered.get("reason", "")),
                ]
            )
        )
    return rows


def process_one_item(
    item: dict[str, Any],
    prompt_builder: PromptBuilder,
    model_engine: QwenModelEngine,
    output_parser: LLMOutputParser,
    max_input_chars: int,
    max_new_tokens: int,
) -> tuple[list[str], str, list[str]]:
    sentence_id = build_sentence_id(item)
    text = preprocess_text(str(item["text"]), max_input_chars)
    prompt = prompt_builder.build_prompt(text)
    generation_kwargs: dict[str, Any] = {}
    if max_new_tokens > 0:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    model_output = model_engine.generate_text(prompt, **generation_kwargs)
    try:
        corrected_entities, filtered_entities = output_parser.parse_and_correct_with_audit(text, model_output)
    except Exception:
        corrected_entities = {}
        filtered_entities = []
    bio_lines = BIOConverter.to_bio_lines(text, corrected_entities)
    sentence_map_row = build_sentence_map_row(-1, sentence_id, item, text)
    audit_rows = build_rules_audit_rows(sentence_id, item, filtered_entities)
    return bio_lines, sentence_map_row, audit_rows


def parse_gpu_ids(gpu_ids: str, num_workers: int) -> list[int]:
    if num_workers <= 0:
        return []
    if gpu_ids.strip():
        parsed: list[int] = []
        for part in gpu_ids.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except Exception:
                continue
        return parsed[:num_workers]
    if not torch.cuda.is_available():
        return []
    return list(range(min(torch.cuda.device_count(), num_workers)))


def worker_loop(
    worker_id: int,
    gpu_id: int | None,
    cfg: dict[str, Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    device = f"cuda:{gpu_id}" if gpu_id is not None else None
    prompt_builder = PromptBuilder(cfg["prompt_config"], cfg["few_shots"])
    model_engine = QwenModelEngine(cfg["model_path"], cfg["prompt_config"], device=device)
    output_parser = LLMOutputParser(cfg["rules_config"])

    while True:
        payload = task_queue.get()
        if payload is None:
            break
        index, item = payload
        sentence_id = build_sentence_id(item)
        text = preprocess_text(str(item["text"]), int(cfg["max_input_chars"]))
        prompt = prompt_builder.build_prompt(text)
        generation_kwargs: dict[str, Any] = {}
        if int(cfg["max_new_tokens"]) > 0:
            generation_kwargs["max_new_tokens"] = int(cfg["max_new_tokens"])
        try:
            model_output = model_engine.generate_text(prompt, **generation_kwargs)
            corrected_entities, filtered_entities = output_parser.parse_and_correct_with_audit(text, model_output)
        except Exception:
            corrected_entities = {}
            filtered_entities = []
        bio_lines = BIOConverter.to_bio_lines(text, corrected_entities)
        sentence_map_row = build_sentence_map_row(index, sentence_id, item, text)
        audit_rows = build_rules_audit_rows(sentence_id, item, filtered_entities)
        result_queue.put(
            {
                "index": index,
                "bio_lines": bio_lines,
                "sentence_map_row": sentence_map_row,
                "audit_rows": audit_rows,
                "source_file": str(item["source_file"]),
                "page_number": str(item["page_number"]),
                "block_index": str(item["block_index"]),
                "worker_id": worker_id,
            }
        )


def write_outputs(args: argparse.Namespace, result_items: list[dict[str, Any]]) -> None:
    result_items.sort(key=lambda x: int(x["index"]))
    all_bio_lines: list[str] = []
    sentence_map_rows: list[str] = [
        "sentence_index\tsentence_id\tsource_file\tsource_path\tpage_number\tblock_index\ttext_length\ttext"
    ]
    rules_audit_rows: list[str] = [
        "sentence_id\tsource_file\tpage_number\tblock_index\tentity_type\tentity_text\treason"
    ]
    for item in result_items:
        all_bio_lines.extend(item["bio_lines"])
        sentence_map_rows.append(item["sentence_map_row"])
        rules_audit_rows.extend(item["audit_rows"])

    BIOConverter.write_bio_file(args.output_file, all_bio_lines)
    sentence_map_path = Path(args.sentence_map_file)
    sentence_map_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_map_path.write_text("\n".join(sentence_map_rows) + "\n", encoding="utf-8")
    rules_audit_path = Path(args.rules_audit_file)
    rules_audit_path.parent.mkdir(parents=True, exist_ok=True)
    rules_audit_path.write_text("\n".join(rules_audit_rows) + "\n", encoding="utf-8")
    print(f"处理完成，共生成 {len(result_items)} 条样本，输出文件: {args.output_file}")
    print(f"sentence 映射文件: {sentence_map_path}")
    print(f"规则审计文件: {rules_audit_path}")


def run_pipeline_sequential(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    prompt_builder = PromptBuilder(args.prompt_config, args.few_shots)
    model_engine = QwenModelEngine(args.model_path, args.prompt_config)
    output_parser = LLMOutputParser(args.rules_config)
    result_items: list[dict[str, Any]] = []

    progress = tqdm(items, total=len(items), desc="处理进度", unit="page_text", dynamic_ncols=True)
    for index, item in enumerate(progress):
        progress.set_postfix(
            file=item["source_file"],
            page=item["page_number"] or "?",
            block=item["block_index"],
            current=f"{index + 1}/{len(items)}",
            refresh=True,
        )
        bio_lines, sentence_map_row, audit_rows = process_one_item(
            item,
            prompt_builder,
            model_engine,
            output_parser,
            args.max_input_chars,
            args.max_new_tokens,
        )
        sentence_map_row = sentence_map_row.replace("-1\t", f"{index}\t", 1)
        result_items.append(
            {
                "index": index,
                "bio_lines": bio_lines,
                "sentence_map_row": sentence_map_row,
                "audit_rows": audit_rows,
            }
        )
    write_outputs(args, result_items)


def run_pipeline_parallel(args: argparse.Namespace, items: list[dict[str, Any]], gpu_ids: list[int]) -> None:
    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue(maxsize=max(64, len(items)))
    result_queue: mp.Queue = ctx.Queue(maxsize=max(64, len(items)))
    cfg = {
        "model_path": args.model_path,
        "prompt_config": args.prompt_config,
        "few_shots": args.few_shots,
        "rules_config": args.rules_config,
        "max_input_chars": args.max_input_chars,
        "max_new_tokens": args.max_new_tokens,
    }

    workers: list[mp.Process] = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        worker = ctx.Process(target=worker_loop, args=(worker_id, gpu_id, cfg, task_queue, result_queue), daemon=True)
        worker.start()
        workers.append(worker)

    for index, item in enumerate(items):
        task_queue.put((index, item))
    for _ in workers:
        task_queue.put(None)

    result_items: list[dict[str, Any]] = []
    received = 0
    progress = tqdm(total=len(items), desc="并行处理进度", unit="page_text", dynamic_ncols=True)
    while received < len(items):
        try:
            result = result_queue.get(timeout=1.0)
        except Empty:
            continue
        received += 1
        progress.set_postfix(
            file=result.get("source_file", ""),
            page=result.get("page_number", ""),
            block=result.get("block_index", ""),
            worker=result.get("worker_id", ""),
            current=f"{received}/{len(items)}",
            refresh=True,
        )
        progress.update(1)
        result_items.append(result)
    progress.close()

    for worker in workers:
        worker.join()

    write_outputs(args, result_items)


def run_pipeline(args: argparse.Namespace) -> None:
    data_loader = DataLoader(args.input_dir)
    items = data_loader.load_items_from_directory()
    if not items:
        write_outputs(args, [])
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids, args.num_workers)
    if args.num_workers > 1 and gpu_ids:
        print(f"启用并行推理，worker={len(gpu_ids)}，gpu_ids={gpu_ids}")
        run_pipeline_parallel(args, items, gpu_ids)
        return

    if args.num_workers > 1 and not gpu_ids:
        print("未检测到可用 GPU 并行配置，回退为单进程模式")
    run_pipeline_sequential(args, items)


if __name__ == "__main__":
    cli_args = parse_args()
    Path(cli_args.output_file).parent.mkdir(parents=True, exist_ok=True)
    run_pipeline(cli_args)
