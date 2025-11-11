from utils.enums import LMModels, Device
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LMEncoder:
    def __init__(self, lm: LMModels, device: Device):
        self.tokenizer = AutoTokenizer.from_pretrained(lm.value)
        self.torch_dtype = torch.float16 if device == Device.CUDA else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            lm.value,
            torch_dtype=self.torch_dtype,
        ).to(device.value)
        self.model.eval()
        self.set_pad_token()

    def set_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token