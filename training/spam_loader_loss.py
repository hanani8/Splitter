from typing import Optional
import torch
from .spam_batch_loss import SpamBatchLoss
from ..models import GPTModel

class SpamLoaderLoss:
    def __init__(self, model: GPTModel, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.device = device
        self.model = model
        self.batch_loss = SpamBatchLoss(model, device)

    def calc(self, data_loader: torch.utils.data.DataLoader, num_batches: Optional[int] = None):
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")

        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                loss = self.batch_loss.calc(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches