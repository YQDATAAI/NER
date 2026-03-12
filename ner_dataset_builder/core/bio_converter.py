from pathlib import Path


class BIOConverter:
    @staticmethod
    def to_bio_lines(text: str, entities: dict[str, dict[str, list[list[int]]]]) -> list[str]:
        chars = list(text)
        tags = ["O"] * len(chars)

        spans: list[tuple[int, int, str]] = []
        for entity_type, entity_map in entities.items():
            for offsets in entity_map.values():
                for start, end in offsets:
                    if 0 <= start <= end < len(chars):
                        spans.append((start, end, entity_type))

        spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        for start, end, entity_type in spans:
            if any(tag != "O" for tag in tags[start : end + 1]):
                continue
            tags[start] = f"B-{entity_type}"
            for idx in range(start + 1, end + 1):
                tags[idx] = f"I-{entity_type}"

        lines = [f"{ch} {tag}" for ch, tag in zip(chars, tags)]
        lines.append("")
        return lines

    @staticmethod
    def write_bio_file(output_path: str, all_lines: list[str]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(all_lines), encoding="utf-8")
