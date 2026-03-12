import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml


TARGET_ENTITY_TYPES = {
    "well_name",
    "block",
    "strat_unit",
    "lithology",
    "lithology_color",
    "lithology_texture",
}


class LLMOutputParser:
    def __init__(self, rules_config_path: str | None = None) -> None:
        self.rules = self._load_rules(rules_config_path)

    @staticmethod
    def _load_rules(rules_config_path: str | None) -> dict[str, dict[str, list[str]]]:
        if not rules_config_path:
            return {}
        path = Path(rules_config_path)
        if not path.exists():
            return {}
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(content, dict):
            return {}
        rules: dict[str, dict[str, list[str]]] = {}
        for entity_type, config in content.items():
            if not isinstance(entity_type, str) or not isinstance(config, dict):
                continue
            include_patterns = config.get("include_patterns", [])
            exclude_texts = config.get("exclude_texts", [])
            valid_patterns = [item for item in include_patterns if isinstance(item, str) and item.strip()]
            valid_excludes = [item for item in exclude_texts if isinstance(item, str) and item.strip()]
            if valid_patterns or valid_excludes:
                rules[entity_type] = {
                    "include_patterns": valid_patterns,
                    "exclude_texts": valid_excludes,
                }
        return rules

    def _check_rules(self, entity_type: str, entity_text: str) -> tuple[bool, str]:
        if not entity_text:
            return False, "empty_entity_text"
        config = self.rules.get(entity_type)
        if not config:
            return True, ""
        exclude_texts = config.get("exclude_texts", [])
        if entity_text in exclude_texts:
            return False, "excluded_text"
        include_patterns = config.get("include_patterns", [])
        if not include_patterns:
            return True, ""
        for pattern in include_patterns:
            try:
                if re.fullmatch(pattern, entity_text):
                    return True, ""
            except re.error:
                continue
        return False, "not_match_include_patterns"

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_json_text(text: str) -> str:
        stripped = LLMOutputParser._strip_markdown_code_fence(text)
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("未找到可解析的 JSON 对象")
        return stripped[start : end + 1]

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        normalized = text.strip()
        normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        return normalized

    def parse_json(self, model_output: str) -> dict[str, Any]:
        json_text = self._extract_json_text(model_output)
        candidates = [json_text, self._normalize_json_text(json_text)]
        last_error: Exception | None = None

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                last_error = exc
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                last_error = exc

        raise ValueError(f"模型输出 JSON 解析失败: {last_error}")

    @staticmethod
    def _find_all_offsets(text: str, entity_text: str) -> list[list[int]]:
        if not entity_text:
            return []
        offsets: list[list[int]] = []
        for match in re.finditer(re.escape(entity_text), text):
            start = match.start()
            end = match.end() - 1
            offsets.append([start, end])
        return offsets

    def correct_offsets(
        self, text: str, parsed_entities: dict[str, Any]
    ) -> tuple[dict[str, dict[str, list[list[int]]]], list[dict[str, str]]]:
        corrected: dict[str, dict[str, list[list[int]]]] = {}
        filtered: list[dict[str, str]] = []
        for entity_type, value in parsed_entities.items():
            if entity_type not in TARGET_ENTITY_TYPES:
                continue
            if not isinstance(value, dict):
                continue
            entity_map: dict[str, list[list[int]]] = {}
            for entity_text in value.keys():
                if not isinstance(entity_text, str):
                    continue
                is_valid, reason = self._check_rules(entity_type, entity_text)
                if not is_valid:
                    filtered.append(
                        {
                            "entity_type": entity_type,
                            "entity_text": entity_text,
                            "reason": reason,
                        }
                    )
                    continue
                real_offsets = self._find_all_offsets(text, entity_text)
                if real_offsets:
                    entity_map[entity_text] = real_offsets
            if entity_map:
                corrected[entity_type] = entity_map
        return corrected, filtered

    def parse_and_correct_with_audit(
        self, text: str, model_output: str
    ) -> tuple[dict[str, dict[str, list[list[int]]]], list[dict[str, str]]]:
        try:
            parsed = self.parse_json(model_output)
        except Exception:
            return {}, []
        return self.correct_offsets(text, parsed)

    def parse_and_correct(self, text: str, model_output: str) -> dict[str, dict[str, list[list[int]]]]:
        corrected, _ = self.parse_and_correct_with_audit(text, model_output)
        return corrected
