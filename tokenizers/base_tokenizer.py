class BaseTokenizer:
    def encode(self, text:str) -> list:
        raise NotImplementedError
    
    def decode(self, token_ids: list) -> str:
        raise NotImplementedError