import os
import torch
from pathlib import Path

# Force tiktoken to use local cache before importing it to prevent DNS NameResolutionErrors
cache_dir = Path(__file__).parent.parent / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir.absolute())
import tiktoken

class PCEITokenizer:
    """Canonical PCEI tokenizer using GPT-2 BPE via tiktoken."""
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_id = 0
        self.eos_id = 50256
    
    def encode(self, text: str, pad: bool = False, max_len: int = 512) -> torch.Tensor:
        raw_ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        ids = [tok_id if tok_id != 0 else (self.vocab_size - 1) for tok_id in raw_ids]
        if pad:
            ids = ids[:max_len]
            while len(ids) < max_len:
                ids.append(self.pad_id)
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            raw_ids = [tok_id.item() for tok_id in ids if tok_id.item() != self.pad_id]
        else:
            raw_ids = [tok_id for tok_id in ids if tok_id != self.pad_id]
        raw_ids = [0 if tok_id == (self.vocab_size - 1) else tok_id for tok_id in raw_ids]
        try:
            return self.enc.decode(raw_ids)
        except Exception:
            return ""
