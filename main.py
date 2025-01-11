from preprocessors import RegexPreprocessor
from tokenizers import SimpleTokenizerV1, SimpleTokenizerV2, TiktokenTokenizer
from datasets import GPTDatasetV1
from dataloaders import create_dataloader
from utils import create_dataloader_v1, create_embedding_layer, generate_text_simple, calc_loss_batch, calc_loss_loader, train_model_simple
from torch import arange
from attention import MultiHeadAttentionWrapper, CausalAttention
from models import DummyGPTModel, GPTModel
import torch
from training import LoaderLoss, BatchLoss, SimpleTrainer

torch.manual_seed(123)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = TiktokenTokenizer()

filepath = "The_Verdict.txt"
with open(filepath, "r", encoding="utf-8") as f:
    text_data = f.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters: ", total_characters)
print("Tokens: ", total_tokens)

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
test_data = text_data[split_idx:]

train_loader = create_dataloader(text = train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M['context_length'], drop_last=True, shuffle=True, num_workers=0)
val_loader = create_dataloader(text = test_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M['context_length'], drop_last=False, shuffle=False, num_workers=0)

print("Loading Model ... ")
model = GPTModel(GPT_CONFIG_124M)
print("Model Loaded")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0004,
    weight_decay = 0.1)

trainer = SimpleTrainer(model=model, train_loader=train_loader, validation_loader=val_loader, optimizer=optimizer, tokenizer=tokenizer)

num_epochs = 10

train_losses, validation_losses, tokens_seen = trainer.train(
    num_epochs = num_epochs,
    eval_freq = 5,
    eval_iter = 5,
    start_context = "I would love to")