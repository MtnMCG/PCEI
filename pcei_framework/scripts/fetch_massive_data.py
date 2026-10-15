import os
import requests
from pathlib import Path

def fetch():
    print("🚀 Initializing Massive Data Fetch...")
    base_dir = Path("data/raw/text")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. WikiText-103 (Stable Mirror)
    print("Fetching WikiText-103...")
    wiki_url = "https://raw.githubusercontent.com/j-min/wikitext-103/master/wiki.train.tokens"
    r = requests.get(wiki_url, stream=True)
    with open(base_dir / "wiki_train.txt", 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            print(".", end="", flush=True)
    print("\nWikiText-103 Downloaded.")

    # 2. Large Gutenberg Subset (Selected Top 100 Books for variety)
    print("Fetching Gutenberg Books...")
    books_dir = base_dir / "gutenberg"
    books_dir.mkdir(exist_ok=True)
    
    # Top 50 books from Gutenberg Project stable links
    # Just a sample to hit the GB target
    book_ids = range(1, 100) 
    for b_id in book_ids:
        try:
            url = f"https://www.gutenberg.org/files/{b_id}/{b_id}-0.txt"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with open(books_dir / f"book_{b_id}.txt", 'w') as f:
                    f.write(r.text)
                print(f"[{b_id}]", end="", flush=True)
        except:
            continue
            
    print("\nFETCH COMPLETE. PCEI FUEL READY.")

if __name__ == "__main__":
    fetch()
