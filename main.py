from preprocessors import RegexPreprocessor
from tokenizers import SimpleTokenizerV1, SimpleTokenizerV2, TiktokenTokenizer
from datasets import GPTDatasetV1

import torch
from torch.utils.data import DataLoader

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = TiktokenTokenizer()
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


path = 'The_Verdict.txt'

with open(path, 'r', encoding='utf8') as file:
    text = file.read()

