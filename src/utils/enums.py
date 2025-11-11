from enum import StrEnum

class LMModels(StrEnum):
    DEFAULT = "bigvul/starcoder2-3b"

class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"