import re

class RegexPreprocessor:
    def __init__(self, regex=r'[,.?_!"()\']|--|\s'):
        self.regex = regex

    def __call__(self, text: str) -> str:
        preprocessed = re.split(self.regex, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return preprocessed