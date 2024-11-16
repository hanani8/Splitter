from preprocessors import RegexPreprocessor
from tokenizers import SimpleTokenizerV1, SimpleTokenizerV2, TiktokenTokenizer
from datasets import GPTDatasetV1
from utils import create_dataloader_v1, create_embedding_layer
from torch import arange

path = 'The_Verdict.txt'

with open(path, 'r', encoding='utf8') as file:
    text = file.read()

    max_length = 4

    tokenizer = TiktokenTokenizer()
    dataset = GPTDatasetV1(text, tokenizer, max_length=max_length, stride=1)
    dataloader = create_dataloader_v1(dataset, batch_size=8, shuffle=False, drop_last=False)

    data_iter = iter(dataloader)
    inputs, target = next(data_iter)
    token_embedding_layer = create_embedding_layer(vocab_size=tokenizer.vocab_size, output_dim=256)
    pos_embedding_layer = create_embedding_layer(vocab_size=max_length, output_dim=256)

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings

    print(input_embeddings.shape)