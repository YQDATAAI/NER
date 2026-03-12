import json
from pathlib import Path
from typing import Any

import yaml


class PromptBuilder:
    def __init__(
        self,
        prompt_config_path: str = "/home/superuser/dev/NER/ner_dataset_builder/configs/prompt_config.json",
        few_shots_path: str = "/home/superuser/dev/NER/ner_dataset_builder/configs/few_shots.yaml",
    ) -> None:
        self.prompt_config = self._load_json(prompt_config_path)
        self.few_shots = self._load_yaml(few_shots_path)
        self.few_shots_str = self.build_few_shots_str()

    @staticmethod
    def _load_json(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        content = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(content)
        return loaded if isinstance(loaded, dict) else {}

    def build_few_shots_str(self) -> str:
        examples = self.few_shots.get("examples", [])
        chunks: list[str] = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            text = str(example.get("text", ""))
            label = str(example.get("label", "{}"))
            chunks.append(f'text: "{text}"\nlabel: {label}')
        return "\n\n".join(chunks)

    @staticmethod
    def _inject_template(template: str, values: dict[str, str]) -> str:
        result = template
        for key, value in values.items():
            result = result.replace("{" + key + "}", value)
        return result

    def build_prompt(self, input_text: str) -> str:
        template = self.prompt_config.get("template", "")
        values = {
            "system_instruction": self.prompt_config.get("system_instruction", ""),
            "task_description": self.prompt_config.get("task_description", ""),
            "few_shots_str": self.few_shots_str,
            "input_text": input_text,
        }
        return self._inject_template(template, values)
