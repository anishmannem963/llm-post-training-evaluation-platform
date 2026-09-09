from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    response: str
    latency_seconds: float
    generated_tokens: int


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseModelRunner:
    def __init__(self, model_name: str, adapter_path: str | None = None) -> None:
        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device)
        self.model.eval()
        self.adapter_path = adapter_path

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

        started = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - started
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated = outputs[0, prompt_tokens:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return GenerationResult(
            response=response,
            latency_seconds=latency,
            generated_tokens=int(generated.shape[-1]),
        )
