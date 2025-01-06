import torch
torch.manual_seed(789)
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # We use register_buffer to make sure that the mask is saved in the state_dict and moved to the device and more importantly it is not a learnable parameter

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)

        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec

        # Information Leakeage from the future is avoided by masking the future tokens and then normalizing the weights
        # This is the Elegance of the Softmax
