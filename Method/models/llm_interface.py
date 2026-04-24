import os
import json
import yaml
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("peft not installed, LoRA fine-tuning will be disabled")

import torch.nn.functional as F


@dataclass
class SpatialRelationResult:
    relation_text: str
    relation_class: int
    confidence: float
    logits: Optional[torch.Tensor] = None


class SpatialLLMInterface(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config['model']['name']
        self.device = torch.device(self.config['model']['device'] if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.vision_projection = None
        self.relation_mapping = self.config['prompt']['relation_mapping']
        self.default_class = self.relation_mapping.get('default', 4)
        self.gen_config = GenerationConfig(
            max_new_tokens=self.config['generation']['max_new_tokens'],
            temperature=self.config['generation']['temperature'],
            top_p=self.config['generation']['top_p'],
            do_sample=self.config['generation']['do_sample'],
            repetition_penalty=self.config['generation']['repetition_penalty'],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        print(f"[LLMInterface] Initialized with model: {self.model_name}")
        print(f"[LLMInterface] Device: {self.device}")
        print(f"[LLMInterface] Trainable: {self.training}")

    def _load_model(self):
        bnb_config = None
        if self.config['model'].get('precision') == 'int8':
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        elif self.config['model'].get('precision') == 'int4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device.type == 'cuda':
            device_map = 'cuda:0'
        else:
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if self.config['model'].get('precision') == 'fp16' else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )

        if self.config['training'].get('freeze_llm', True):
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        if self.config['training'].get('use_lora', False) and PEFT_AVAILABLE:
            self._setup_lora()

    def _setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['training']['lora_r'],
            lora_alpha=self.config['training']['lora_alpha'],
            lora_dropout=self.config['training']['lora_dropout'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"[LLMInterface] LoRA enabled: r={self.config['training']['lora_r']}")

    def _build_prompt(self, bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> str:
        cx1, cy1, w1, h1 = bbox1
        cx2, cy2, w2, h2 = bbox2
        template = self.config['prompt']['user_template']
        prompt = template.format(
            cx1=cx1, cy1=cy1, w1=w1, h1=h1,
            cx2=cx2, cy2=cy2, w2=w2, h2=h2
        )
        return prompt

    def _build_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        system_prompt = self.config['prompt']['system']
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    def _map_relation_to_class(self, relation_text: str) -> int:
        text = relation_text.lower().strip().strip('.')
        if text in self.relation_mapping:
            return self.relation_mapping[text]
        horizontal = 1
        vertical = 1
        if any(word in text for word in ['left', 'left of', 'west']):
            horizontal = 0
        elif any(word in text for word in ['right', 'right of', 'east']):
            horizontal = 2
        if any(word in text for word in ['above', 'top', 'upper', 'north', 'over']):
            vertical = 0
        elif any(word in text for word in ['below', 'bottom', 'lower', 'south', 'under']):
            vertical = 2
        return vertical * 3 + horizontal

    def _compute_confidence(self, outputs) -> float:
        if hasattr(outputs, 'scores') and outputs.scores:
            probs = torch.softmax(outputs.scores[0], dim=-1)
            max_prob = torch.max(probs, dim=-1).values.mean().item()
            return max_prob
        return 0.9

    def predict_spatial_relation(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
        feat1: Optional[torch.Tensor] = None,
        feat2: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> SpatialRelationResult:
        prompt = self._build_prompt(bbox1, bbox2)
        messages = self._build_chat_messages(prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                generation_config=self.gen_config,
                return_dict_in_generate=True,
                output_scores=return_logits,
            )
        generated_ids = outputs.sequences[0][inputs.shape[1]:]
        relation_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        relation_class = self._map_relation_to_class(relation_text)
        confidence = self._compute_confidence(outputs) if return_logits else 0.95
        logits = None
        if return_logits and hasattr(outputs, 'scores'):
            logits = torch.stack(outputs.scores, dim=1)
        return SpatialRelationResult(
            relation_text=relation_text,
            relation_class=relation_class,
            confidence=confidence,
            logits=logits
        )

    def batch_predict_spatial_relations(
        self,
        bbox_pairs: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        feats: Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = None,
        batch_size: int = 4
    ) -> List[SpatialRelationResult]:
        results = []
        for b1, b2 in bbox_pairs:
            results.append(self.predict_spatial_relation(b1, b2))
        return results

    def get_spatial_relation_loss(
        self,
        bbox1: torch.Tensor,
        bbox2: torch.Tensor,
        target_ids: torch.Tensor,
        feat1: Optional[torch.Tensor] = None,
        feat2: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[SpatialRelationResult]]]:
        batch_size = bbox1.shape[0]
        loss = torch.tensor(0.0, device=bbox1.device)
        results = []
        for i in range(batch_size):
            b1 = tuple(bbox1[i].cpu().numpy())
            b2 = tuple(bbox2[i].cpu().numpy())
            result = self.predict_spatial_relation(b1, b2)
            results.append(result)
            pred_class = torch.tensor(result.relation_class, device=bbox1.device)
            target_class = target_ids[i]
            if pred_class != target_class:
                loss = loss + 1.0
        loss = loss / batch_size
        if return_details:
            return loss, results
        return loss

    def set_training_mode(self, mode: bool):
        self.training = mode
        if self.config['training'].get('freeze_llm', True):
            self.model.eval()
        else:
            self.model.train() if mode else self.model.eval()

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[LLMInterface] Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: str):
        instance = cls(config_path)
        instance.model = AutoModelForCausalLM.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return instance


def compute_spatial_loss_with_llm(
    llm_interface: SpatialLLMInterface,
    bbox_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    pred_logits: torch.Tensor,
    target_relations: Optional[List[int]] = None,
    use_llm_labels: bool = True
) -> torch.Tensor:
    device = pred_logits.device
    if use_llm_labels and target_relations is None:
        llm_labels = []
        for bbox_a, bbox_b in bbox_pairs:
            b1 = tuple(bbox_a.cpu().numpy())
            b2 = tuple(bbox_b.cpu().numpy())
            result = llm_interface.predict_spatial_relation(b1, b2)
            llm_labels.append(result.relation_class)
        target = torch.tensor(llm_labels, device=device)
    else:
        target = torch.tensor(target_relations, device=device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(pred_logits, target)
    return loss