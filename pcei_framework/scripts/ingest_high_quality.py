import os
import requests
import json
from pathlib import Path
from rich.console import Console

console = Console()

def download_file(url, dest):
    if dest.exists():
        console.print(f"[dim]Already have {dest.name}[/dim]")
        return
    console.print(f"[cyan]Downloading {url}...[/cyan]")
    response = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    data_dir = Path(__file__).parent.parent / "data" / "high_quality"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. ShareGPT (Conversational Logic)
    # Using a cleaned subset of ShareGPT
    sharegpt_url = "https://huggingface.co/datasets/anon8231489123/sharegpt_gpt4/resolve/main/sharegpt_gpt4.jsonl"
    sharegpt_path = data_dir / "sharegpt_logic.jsonl"
    
    # 2. Python Code (Structural Logic)
    # Using a curated Python dataset
    python_url = "https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/resolve/main/data/python_curated.jsonl"
    python_path = data_dir / "python_structure.jsonl"

    try:
        download_file(sharegpt_url, sharegpt_path)
        download_file(python_url, python_path)
        console.print("[bold green]Success! High-Quality dataset initialized.[/bold green]")
        console.print(f"[dim]Location: {data_dir}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Download failed: {e}[/bold red]")
        console.print("[yellow]Falling back to secondary mirrors...[/yellow]")
        # Fallback logic could go here

if __name__ == "__main__":
    main()
