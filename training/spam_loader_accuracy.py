import torch
from ..models import GPTModel
from typing import Optional

class SpamAccuracyLoader:
    def __init__(self, model: GPTModel, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.device = device
        self.model = model

    def calc(self, data_loader: torch.utils.data.DataLoader, num_batches: Optional[int] = None):
        self.model.eval()

        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch.to(self.device)
                target_batch.to(self.device)

                with torch.no_grad():
                    logits = self.model(input_batch)[:, -1, :]
                
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]

                correct_predictions += {
                    (predicted_labels == target_batch).sum().item()
                }
            
            else:
                break

        return correct_predictions / num_examples