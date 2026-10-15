import json
import re
from pathlib import Path

input_path = Path("/Users/emerson/Desktop/PCEI /hk_lora_chatml_cleaned_2026-06-05.jsonl")
output_path = Path("/Users/emerson/Desktop/PCEI /pcei_framework/data/synthetic/hk_expanded.jsonl")

output_path.parent.mkdir(parents=True, exist_ok=True)

samples = []
original_records = 0

with open(input_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
        except:
            continue
            
        msgs = data.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        
        if not (sys_msg and user_msg and asst_msg): 
            continue
            
        original_records += 1
        
        # Split on "Response: " to break up tool interactions
        parts = re.split(r'\nResponse:\s*', asst_msg)
        
        current_prompt = f"SYSTEM: {sys_msg}\nUSER: {user_msg}\nASSISTANT:"
        
        if len(parts) == 1:
            # No tool calls made
            samples.append({
                "prompt": current_prompt, 
                "response": f" {parts[0].strip()}\n<|endoftext|>"
            })
            continue

        for i, part in enumerate(parts):
            if i == 0:
                # The initial thought and first tool call
                response_target = f" {part.strip()}"
                samples.append({
                    "prompt": current_prompt, 
                    "response": response_target
                })
                current_prompt += response_target + "\nResponse: "
            else:
                # Part contains the previous tool's JSON response, then a newline, then the next assistant generation
                sub_parts = part.split("\n", 1)
                tool_res = sub_parts[0].strip()
                current_prompt += f"{tool_res}\n\n"
                
                asst_gen = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                
                if i == len(parts) - 1:
                    # The final generation block (the actual answer to the user)
                    response_target = f"{asst_gen}\n<|endoftext|>"
                    samples.append({
                        "prompt": current_prompt, 
                        "response": response_target
                    })
                else:
                    # Intermediate block (another tool call)
                    if asst_gen:
                        samples.append({
                            "prompt": current_prompt, 
                            "response": f"{asst_gen}"
                        })
                        current_prompt += f"{asst_gen}\nResponse: "
                    else:
                        current_prompt += "Response: "

print(f"Original Records Processed: {original_records}")
print(f"Total Expanded Training Samples: {len(samples)}")

with open(output_path, 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
        
print(f"Saved expanded dataset to {output_path}")
