import re
from preprocessors import RegexPreprocessor
from tokenizers import BaseTokenizer

class SimpleTokenizerV1(BaseTokenizer):
    def __init__(self, corpus: str, preprocessor: RegexPreprocessor):
        self.preprocessor = preprocessor
        tokens = preprocessor(corpus)
        self.str_to_int = self.create_vocab(tokens)
        self.int_to_str = {id_: token for token, id_ in self.str_to_int.items()}
        self.vocab_size = len(self.str_to_int)

    def encode(self, text:str) -> list:
        preprocessed_text = self.preprocessor(text)
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
        vocab = {token: id_ for id_, token in enumerate(unique_tokens)}
        return vocab
