import re
from preprocessors import RegexPreprocessor
from tokenizers import BaseTokenizer

class SimpleTokenizerV2(BaseTokenizer):
    def __init__(self, corpus: str, preprocessor: RegexPreprocessor, unk_token='<|unk|>'):
        self.preprocessor = preprocessor
        self.unk_token = unk_token
        tokens = preprocessor(corpus)

        self.str_to_int = self.create_vocab(tokens)
        self.int_to_str = {id_: token for token, id_ in self.str_to_int.items()}
        self.vocab_size = len(self.str_to_int)

    def encode(self, text:str) -> list:
        preprocessed_text = self.preprocessor(text)
        preprocessed_text = [token if token in self.str_to_int else self.unk_token for token in preprocessed_text]
        token_ids = [self.str_to_int[token] for token in preprocessed_text]
        return token_ids

    def decode(self, token_ids: list) -> str:
        # Accept token_ids in list format, and convert to text
        text = " ".join([self.int_to_str[id_] for id_ in token_ids])
        text = re.sub(r'\s([.,!?_"](?:\s|$))', r'\1', text)
        return text
    
    def create_vocab(self, tokens: list) -> dict:
        # Create a vocabulary from the tokens
        unique_tokens = sorted(list(set(tokens)))
        unique_tokens.extend([self.unk_token, '<|endoftext|>'])
        vocab = {token: id_ for id_, token in enumerate(unique_tokens)}
        return vocab
