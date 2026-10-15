import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_prompt_formatting_data(num_examples: int = 2000, output_path: Path = None):
    """
    Generates structured 'Prompt: [...] Response: [...]' data so the model
    learns the mapping between a query and an answer format.
    """
    if output_path is None:
        output_path = DATA_DIR / "prompt_formatting.jsonl"
        
    records = []
    
    # Templates that teach the model how to respond
    templates = [
        ("What is the sum of {a} and {b}?", "The sum is {ans}."),
        ("Identify the object: {obj}", "The object is a {obj}."),
        ("Translate {word} to uppercase.", "UPPERCASE: {upper}"),
        ("Repeat this sequence: {seq}", "Sequence: {seq}"),
        ("Is {num} even or odd?", "The number {num} is {parity}.")
    ]
    
    for _ in range(num_examples):
        t_q, t_a = random.choice(templates)
        
        if "sum" in t_q:
            a, b = random.randint(1, 50), random.randint(1, 50)
            q = t_q.format(a=a, b=b)
            a_str = t_a.format(ans=a+b)
        elif "object" in t_q:
            obj = random.choice(["tree", "car", "dog", "cat", "house", "mountain", "river", "bridge"])
            q = t_q.format(obj=obj)
            a_str = t_a.format(obj=obj)
        elif "uppercase" in t_q:
            word = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3, 8)))
            q = t_q.format(word=word)
            a_str = t_a.format(upper=word.upper())
        elif "sequence" in t_q:
            seq = ' '.join(str(random.randint(0, 9)) for _ in range(3))
            q = t_q.format(seq=seq)
            a_str = t_a.format(seq=seq)
        elif "even or odd" in t_q:
            num = random.randint(1, 100)
            parity = "even" if num % 2 == 0 else "odd"
            q = t_q.format(num=num)
            a_str = t_a.format(num=num, parity=parity)
            
        text = f"Prompt: {q}\nResponse: {a_str}"
        records.append({"text": text, "type": "prompt_format"})
        
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(records)} prompt formatting examples at {output_path}")

if __name__ == "__main__":
    generate_prompt_formatting_data()
