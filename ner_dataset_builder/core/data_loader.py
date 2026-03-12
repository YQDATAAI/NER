import json
import re
from pathlib import Path
from typing import Any


class DataLoader:
    def __init__(self, data_dir: str = "/home/superuser/dev/NER/data") -> None:
        self.data_dir = Path(data_dir)

    @staticmethod
    def _parse_json_content(raw_content: str) -> list[dict[str, Any]]:
        parsed = json.loads(raw_content)
        if not isinstance(parsed, list):
            raise ValueError("文件内容不是 JSON 数组")
        return [item for item in parsed if isinstance(item, dict)]

    @staticmethod
    def _remove_table_blocks(text: str) -> str:
        cleaned = re.sub(r"<table\b[^>]*>[\s\S]*?</table>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?table\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def load_texts_from_file(self, file_path: str | Path) -> list[str]:
        items = self.load_items_from_file(file_path)
        return [item["text"] for item in items]

    def load_items_from_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        raw_content = path.read_text(encoding="utf-8")
        records = self._parse_json_content(raw_content)
        items: list[dict[str, Any]] = []
        for block_index, item in enumerate(records):
            page_text = item.get("page_text")
            if not isinstance(page_text, str):
                continue
            stripped = self._remove_table_blocks(page_text).strip()
            if stripped:
                page_number = item.get("page_number")
                items.append(
                    {
                        "source_file": path.name,
                        "source_path": str(path),
                        "page_number": str(page_number) if page_number is not None else "",
                        "block_index": block_index,
                        "text": stripped,
                    }
                )
        return items

    def load_texts_from_directory(self) -> list[str]:
        items = self.load_items_from_directory()
        return [item["text"] for item in items]

    def load_items_from_directory(self) -> list[dict[str, Any]]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        all_items: list[dict[str, Any]] = []
        for file_path in sorted(self.data_dir.glob("*.md")):
            try:
                all_items.extend(self.load_items_from_file(file_path))
            except Exception:
                continue
        return all_items
