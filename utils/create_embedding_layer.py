from torch import nn

def create_embedding_layer(output_dim: int, vocab_size: int) -> nn.Embedding:
    embedding = nn.Embedding(vocab_size, output_dim)
    return embedding