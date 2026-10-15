import os
import tiktoken

cache_dir = os.path.abspath("./tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

print("Downloading gpt2 encoding to local cache:", cache_dir)
enc = tiktoken.get_encoding("gpt2")
print("Done. Cached files:")
for f in os.listdir(cache_dir):
    print(" -", f)
