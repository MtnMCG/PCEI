import json
import argparse
import os
from pathlib import Path

def convert_chatml_to_pcei(input_path, output_path):
    print(f"Reading ChatML from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    converted = []
    for line in lines:
        if not line.strip(): continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        if "messages" in data:
            prompt_str = ""
            response_str = ""
            for msg in data["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt_str += f"SYSTEM: {content}\\n"
                elif role == "user":
                    prompt_str += f"USER: {content}\\n"
                elif role == "assistant":
                    response_str = f" {content}\\n<|endoftext|>"
            
            if prompt_str and response_str:
                # Format to exact PCEI synthetic data specifications
                pcei_entry = {
                    "prompt": prompt_str.strip() + "\\nASSISTANT:",
                    "response": response_str,
                    "used": False
                }
                converted.append(json.dumps(pcei_entry))
                
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for c in converted:
            f.write(c + "\n")
            
    print(f"Successfully converted and appended {len(converted)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import standard ChatML jsonl into PCEI synthetic data format.")
    parser.add_argument("input_file", type=str, help="Path to input .jsonl file (e.g. hk_lora_chatml_cleaned.jsonl)")
    parser.add_argument("--output", type=str, default="data/synthetic/instruction_tuning.jsonl", help="Path to output PCEI data file")
    args = parser.parse_args()
    
    convert_chatml_to_pcei(args.input_file, args.output)
