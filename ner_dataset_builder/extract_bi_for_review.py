import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 BIO 文件中提取 B/I 标签内容供人工复核")
    parser.add_argument(
        "--input-file",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.bio",
        help="输入 BIO 文件路径",
    )
    parser.add_argument(
        "--bi-lines-output",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.bi_only.tsv",
        help="导出逐字符 B/I 行的文件路径",
    )
    parser.add_argument(
        "--spans-output",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.bi_spans.tsv",
        help="导出实体片段文件路径",
    )
    parser.add_argument(
        "--sentence-map-file",
        default="/home/superuser/dev/NER/ner_dataset_builder/output/train.sentence_map.tsv",
        help="sentence_id 映射文件路径",
    )
    return parser.parse_args()


def load_sentence_meta(sentence_map_path: Path) -> list[tuple[str, int]]:
    if not sentence_map_path.exists():
        return []
    raw_lines = sentence_map_path.read_text(encoding="utf-8").splitlines()
    sentence_meta: list[tuple[str, int]] = []
    for idx, line in enumerate(raw_lines):
        if idx == 0:
            continue
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        sentence_id = parts[1]
        try:
            text_length = int(parts[6])
        except Exception:
            text_length = -1
        sentence_meta.append((sentence_id, text_length))
    return sentence_meta


def resolve_sentence_id(sentence_index: int, sentence_meta: list[tuple[str, int]]) -> str:
    if 0 <= sentence_index < len(sentence_meta):
        return sentence_meta[sentence_index][0]
    return f"sentence_{sentence_index:06d}"


def parse_bio_tokens(raw_text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    idx = 0
    n = len(raw_text)
    while idx < n:
        if raw_text[idx] == "\n" and not (idx + 1 < n and raw_text[idx + 1] == " "):
            idx += 1
            continue
        if idx + 1 >= n:
            break
        ch = raw_text[idx]
        idx += 1
        if idx >= n or raw_text[idx] != " ":
            while idx < n and raw_text[idx] != "\n":
                idx += 1
            if idx < n:
                idx += 1
            continue
        idx += 1
        end = raw_text.find("\n", idx)
        if end == -1:
            tag = raw_text[idx:]
            idx = n
        else:
            tag = raw_text[idx:end]
            idx = end + 1
        if tag:
            tokens.append((ch, tag))
    return tokens


def split_tokens_by_sentence(tokens: list[tuple[str, str]], sentence_meta: list[tuple[str, int]]) -> list[list[tuple[str, str]]]:
    if not sentence_meta:
        return [tokens]
    if any(length < 0 for _, length in sentence_meta):
        return [tokens]
    expected_total = sum(length for _, length in sentence_meta)
    if expected_total > len(tokens):
        return [tokens]
    chunks: list[list[tuple[str, str]]] = []
    start = 0
    for _, length in sentence_meta:
        end = start + length
        chunks.append(tokens[start:end])
        start = end
    return chunks


def extract_bi(input_path: Path, bi_lines_path: Path, spans_path: Path, sentence_meta: list[tuple[str, int]]) -> tuple[int, int]:
    raw_text = input_path.read_text(encoding="utf-8")
    tokens = parse_bio_tokens(raw_text)
    sentence_chunks = split_tokens_by_sentence(tokens, sentence_meta)

    bi_rows: list[str] = ["sentence_id\ttoken_idx\tchar\ttag"]
    span_rows: list[str] = ["sentence_id\tentity_type\tstart_idx\tend_idx\ttext"]
    bi_count = 0
    span_count = 0

    for sentence_index, chunk in enumerate(sentence_chunks):
        sentence_key = resolve_sentence_id(sentence_index, sentence_meta)
        current_type = ""
        current_start = -1
        current_chars: list[str] = []

        def flush_span(end_idx: int) -> None:
            nonlocal current_type, current_start, current_chars, span_count
            if not current_type or current_start < 0 or not current_chars:
                current_type = ""
                current_start = -1
                current_chars = []
                return
            text = "".join(current_chars)
            span_rows.append(f"{sentence_key}\t{current_type}\t{current_start}\t{end_idx}\t{text}")
            span_count += 1
            current_type = ""
            current_start = -1
            current_chars = []

        for token_idx, (char, tag) in enumerate(chunk):
            if tag.startswith(("B-", "I-")):
                bi_rows.append(f"{sentence_key}\t{token_idx}\t{char}\t{tag}")
                bi_count += 1

            if tag.startswith("B-"):
                flush_span(token_idx - 1)
                current_type = tag[2:]
                current_start = token_idx
                current_chars = [char]
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_type == entity_type and current_start >= 0:
                    current_chars.append(char)
                else:
                    flush_span(token_idx - 1)
                    current_type = entity_type
                    current_start = token_idx
                    current_chars = [char]
            else:
                flush_span(token_idx - 1)

        flush_span(len(chunk) - 1)

    bi_lines_path.parent.mkdir(parents=True, exist_ok=True)
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    bi_lines_path.write_text("\n".join(bi_rows) + "\n", encoding="utf-8")
    spans_path.write_text("\n".join(span_rows) + "\n", encoding="utf-8")
    return bi_count, span_count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    bi_lines_path = Path(args.bi_lines_output)
    spans_path = Path(args.spans_output)
    sentence_map_path = Path(args.sentence_map_file)
    sentence_meta = load_sentence_meta(sentence_map_path)
    bi_count, span_count = extract_bi(input_path, bi_lines_path, spans_path, sentence_meta)
    print(f"已导出 B/I 行: {bi_count} -> {bi_lines_path}")
    print(f"已导出实体片段: {span_count} -> {spans_path}")
    if sentence_meta:
        print(f"已加载 sentence_id 映射: {len(sentence_meta)} 条 -> {sentence_map_path}")


if __name__ == "__main__":
    main()
