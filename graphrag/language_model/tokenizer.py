import tiktoken
from autotiktokenizer import AutoTikTokenizer
import threading
from typing import Optional


class SingletonTokenizer:
    _instance: Optional['SingletonTokenizer'] = None
    _lock = threading.Lock()

    def __new__(cls, encoding_model: str) -> 'SingletonTokenizer':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    try:
                        instance.encoding = tiktoken.get_encoding(encoding_model)
                    except:
                        instance.encoding = AutoTikTokenizer.from_pretrained(encoding_model)
                    cls._instance = instance
        return cls._instance

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.encoding.encode(text, **kwargs)
    
    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)