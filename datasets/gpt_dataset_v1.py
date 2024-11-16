import torch
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        # Subtraction of max_length is to ensure that the last target sequence is a full
        for i in range(0, len(text) - max_length, stride):
            input_sequence = token_ids[i:i + max_length]
            target_sequence = token_ids[i + 1:i + 1 + max_length]

            self.input_ids.append(input_sequence)
            self.target_ids.append(target_sequence)
    

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])