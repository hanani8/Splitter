from preprocessors import RegexPreprocessor
from tokenizers import SimpleTokenizerV1, SimpleTokenizerV2, TiktokenTokenizer
from datasets import GPTDatasetV1
from utils import create_dataloader_v1, create_embedding_layer, generate_text_simple
from torch import arange
from attention import MultiHeadAttentionWrapper, CausalAttention
from models import DummyGPTModel, GPTModel
import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = TiktokenTokenizer()
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
# out = model(batch)

# print("Input Batch:\n", batch)
# print("Output Shape:\n", out.shape)
# print(out)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("Encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("Encoded Tensor Shape: ", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output: ", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print("Decoded Text: ", decoded_text)