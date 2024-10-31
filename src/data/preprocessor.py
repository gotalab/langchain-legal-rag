from typing import List
from sudachipy import tokenizer, dictionary

def preprocess_func(text: str) -> List[str]:
    """日本語テキストのトークン化を行う"""
    tokenizer_obj = dictionary.Dictionary(dict="core").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text, mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))  # 重複削除
    return words 