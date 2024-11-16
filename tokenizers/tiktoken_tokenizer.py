import tiktoken
from tokenizers import BaseTokenizer

class TiktokenTokenizer(BaseTokenizer):
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, token_ids: list) -> str:
        return self.tokenizer.decode(token_ids)