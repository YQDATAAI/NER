import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenModelEngine:
    def __init__(
        self,
        model_path: str = "/home/superuser/LLM_Model/Qwen3.5-4B",
        prompt_config_path: str = "/home/superuser/dev/NER/ner_dataset_builder/configs/prompt_config.json",
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.prompt_config = self._load_prompt_config(prompt_config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_device_map: str | dict[str, str]
        if device:
            model_device_map = {"": device}
        else:
            model_device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=model_device_map,
            dtype=self._resolve_dtype(),
            trust_remote_code=True,
        )
        self.model.eval()

    @staticmethod
    def _load_prompt_config(prompt_config_path: str) -> dict[str, Any]:
        config_path = Path(prompt_config_path)
        return json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _resolve_dtype() -> torch.dtype:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        generation_config = self.prompt_config.get("generation_config", {})
        merged_config = {**generation_config, **kwargs}
        merged_config = self._sanitize_generation_config(merged_config)

        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        target_device = self.model.device
        model_inputs = {k: v.to(target_device) for k, v in model_inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**model_inputs, **merged_config)

        input_len = model_inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _sanitize_generation_config(self, config: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(config)
        temperature = sanitized.get("temperature")
        top_p = sanitized.get("top_p")
        do_sample = bool(sanitized.get("do_sample", False))
        do_sample_explicit = "do_sample" in sanitized

        if temperature is not None:
            try:
                temperature = float(temperature)
            except Exception:
                temperature = None
            if temperature is None or temperature <= 0:
                sanitized.pop("temperature", None)
            elif not do_sample_explicit:
                do_sample = True

        if top_p is not None:
            try:
                top_p = float(top_p)
            except Exception:
                top_p = None
            if top_p is None or top_p <= 0 or top_p >= 1:
                sanitized.pop("top_p", None)
            elif not do_sample_explicit and temperature is not None and temperature > 0:
                do_sample = True

        if do_sample:
            sanitized["do_sample"] = True
        else:
            sanitized.pop("temperature", None)
            sanitized.pop("top_p", None)
            sanitized["do_sample"] = False

        if self.tokenizer.eos_token_id is not None:
            sanitized.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        return sanitized
