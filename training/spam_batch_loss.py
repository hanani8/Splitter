from typing import Optional
import torch
from ..models import GPTModel

class SpamBatchLoss:
    def __init__(self, model: GPTModel, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.model = model
        self.device = device

    def calc(self, input_batch: torch.tensor, target_batch: torch.tensor):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss