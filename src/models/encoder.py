from utils.enums import LMModels, Device
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LMEncoder:
    def __init__(self, lm: LMModels, device: Device, max_length: int = 512):
        self.device = device
        self.torch_dtype = torch.float16 if device == Device.CUDA else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(lm.value)
        self.model = AutoModelForCausalLM.from_pretrained(
            lm.value, torch_dtype=self.torch_dtype
        ).to(device.value)
        self.model.eval()
        self.max_tokenizer_length = max_length
        self._set_pad_token()

    def _set_pad_token(self) -> None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, code_snippet: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_tokenizer_length,
        )
        input_ids = enc["input_ids"].to(self.device.value)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device.value)
        return input_ids, attn

    def compute_entropy(self, code_snippet: str) -> dict:

        input_ids, attn = self._encode(code_snippet)

        # prepare labels that ignore pad tokens in loss (set to -100)
        labels = input_ids.clone()
        labels[attn == 0] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attn, labels=labels)

        # compute per-token (shifted) loss to get sum_entropy and per-token breakdown
        shift_logits = outputs.logits[:, :-1, :] # [B, T-1, V]
        shift_labels = labels[:, 1:]             # [B, T-1]
        attn_shift = attn[:, 1:]                 # [B, T-1]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)                          # [B, T-1]
        ).view_as(shift_labels) * attn_shift.to(torch.float)  # zero out positions that were padding / ignored

        sum_entropy = float(per_token.sum().cpu())
        n_tokens = int(attn_shift.sum().cpu())
        # safe mean computed from per-token sum and token count (avoid divide-by-zero)
        mean_entropy_safe = sum_entropy / n_tokens if n_tokens > 0 else float("nan")

        return {
            "mean_entropy": mean_entropy_safe,
            "sum_entropy": sum_entropy,
            "n_tokens": n_tokens,
            "per_token": per_token.squeeze(0).cpu().tolist()
        }
    
    def compute_embeddings(self, code_snippet: str) -> dict:
        input_ids, attn = self._encode(code_snippet)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attn, output_hidden_states=True)
            # For causal LMs, the last hidden state is in outputs.hidden_states[-1]
            # Shape: [batch_size, seq_len, hidden_size]
            embeddings = outputs.hidden_states[-1].squeeze(0)  # [seq_len, hidden_size]
            pooled = embeddings.mean(dim=0)
    
        return {
            "embeddings": pooled.cpu(),
            "input_ids": input_ids.squeeze(0).cpu(),
            "attention_mask": attn.squeeze(0).cpu()
        }