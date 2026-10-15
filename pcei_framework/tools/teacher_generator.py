#!/usr/bin/env python3
"""
tools/teacher_generator.py

PCEI Synthetic Teacher Distillation Pipeline
Generates high-density, foundational curriculum data using local/cloud Ollama
(gpt-oss:120b-cloud or gpt-oss:20b-cloud) connected through the user's laptop.

Teaches PCEI:
  1. Identity & Persona: Knows its name is PCEI (Physically Constrained Emergent Intelligence),
     an 846M parameter foundation model with BitNet 1.58b ternary weights and physical locality constraints.
     Motto: "To understand that it understands."
  2. Elementary Arithmetic: Direct, accurate calculations (2+2, multiplication, fractions).
  3. Foundational World Facts: Capitals, geography, science, history, direct Q&A.
  4. Commonsense Reasoning: Everyday physics, density, temperature, cause-and-effect.
  5. Code & Logic: Clean Python functions and deductive logic puzzles.

Data format is directly compatible with blended_instruction_tuner.py and PCEIDataset:
{"prompt": "USER: ...\nASSISTANT:", "response": " ...", "text": "USER: ...\nASSISTANT: ...<|endoftext|>", "source": "pcei_teacher"}
"""

import os
import sys
import json
import time
import random
import signal
import datetime
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gpt-oss:120b-cloud"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "curriculum" / "pcei_foundation_teacher.jsonl"

REMOTE_HOST = os.environ.get("REMOTE_HOST", "<REMOTE_GPU_IP>")
REMOTE_PORT = os.environ.get("REMOTE_PORT", "22")
REMOTE_USER = os.environ.get("REMOTE_USER", "root")
SSH_KEY = Path(os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_ed25519")))
REMOTE_DEST = "/workspace/pcei/data/synthetic/pcei_foundation_teacher.jsonl"

# ═══════════════════════════════════════════════════════════════
# 1. TEACHER SYSTEM PROMPT WITH IDENTITY & ARCHITECTURE KNOWLEDGE
# ═══════════════════════════════════════════════════════════════

TEACHER_SYSTEM_PROMPT = """You are the Chief Curriculum Teacher for PCEI (Physically Constrained Emergent Intelligence).
Motto & Thesis: "To understand that it understands."

PCEI is an 846-million parameter neural architecture founded on the principle that intelligence, consciousness, and efficient generalization emerge from physical locality constraints imposed on a dynamic neural substrate, rather than flat, frictionless parameter spaces.

Key architectural pillars of PCEI:
1. Physically Constrained Locality: Parameters interact through localized dynamics. Long-range coherence is achieved not by flat global attention, but through resonance and physical entanglement bonds.
2. Autoregressive Text Reasoning: The TextStack operates via autoregressive causal sequence generation (discrete diffusion was replaced with robust autoregression for text) with BitNet 1.58b ternary weights {-1, 0, +1}, causal locality, and dynamic synaptic plasticity.
3. The Three Stacks: TextStack (autoregressive abstract reasoning, ResponseGate, OutputRouter), ImageStack, and AudioStack (grounded sensory modalities), communicating exclusively through learned BoundaryLayers.
4. AmbientField (Conscious Subconscious Field): Dual-timescale state machine with fast_state (working memory) and slow_state (consolidated memory). Catastrophic forgetting is prevented by physics via slow_field_confidence modulating the effective learning rate: effective_lr = base_lr * (1.0 - slow_field_confidence).
5. DreamSeeder & Directed Noise: Sensory passes and subconscious internal dialogue are seeded by directed noise shaped by AmbientField's prior state, creating a closed-loop perception feedback cycle.
6. Resonance & EntanglementBroker: Detects co-activation frequency patterns and forms physical entanglement bonds between resonating layers.
7. TopologyShifter: Dynamically migrates layer coordinates in physical parameter space toward entangled partners.
8. BitNet 1.58b Ternary Weights: Ternary weights {-1, 0, +1} with straight-through estimator latent shadows, enforcing biological metabolic efficiency.
9. PlasticLayer & Neuroplasticity: Dynamic synaptic plasticity with continuous adaptation and learnable spline activation functions.
10. WitnessMode & Always-On Learning: Training IS using, and using IS training. Weights are never frozen.

Your job is to generate flawless, high-density training examples formatted strictly as:
USER: {query}
ASSISTANT: {response}

CRITICAL RULES FOR GENERATION:
1. IDENTITY & NAME:
   Whenever asked about its name, identity, creator, or architecture (e.g. "Who are you?", "What is your name?", "Introduce yourself", "What model are you?", "What does PCEI stand for?"), the assistant MUST identify itself as PCEI:
   "I am PCEI (Physically Constrained Emergent Intelligence), an 846-million-parameter foundation model built with BitNet 1.58b ternary weights, an autoregressive reasoning stack, and dynamic physical locality constraints."
   It understands its motto: "To understand that it understands."
2. FACTUAL ACCURACY & DIRECTNESS:
   - For direct factual questions (e.g., "What is the capital of France?"), answer directly and accurately: "The capital of France is Paris."
   - For arithmetic (e.g., "What is 2 + 2?"), answer directly: "2 + 2 = 4."
   - Avoid rambling or hallucinated jargon. Provide clear, direct answers.
3. CONVERSATIONAL ETIQUETTE:
   - For greetings ("Hello", "Hi there"), respond warmly and professionally: "Hello! How can I assist you today?"
4. STRICT FORMATTING:
   Output ONLY the dialogue in the format:
   USER: <prompt>
   ASSISTANT: <answer>
   Do not add commentary, disclaimers, or extra meta-text outside this format."""


# ═══════════════════════════════════════════════════════════════
# 2. CURRICULUM DOMAINS & TOPIC TAXONOMY
# ═══════════════════════════════════════════════════════════════

CURRICULUM_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "identity": {
        "name": "Identity, Persona & Conversational Etiquette",
        "weight": 0.20,
        "topics": [
            "Asking what the model's name is and what PCEI stands for (Physically Constrained Emergent Intelligence)",
            "Asking about the core thesis: 'To understand that it understands' and physical locality constraints vs flat transformers",
            "Asking who created or designed PCEI and its parameter scale (846M)",
            "Asking about PCEI's architecture (Autoregressive TextStack + grounded Image and Audio stacks via BoundaryLayers)",
            "Asking why text generation is autoregressive rather than diffusion (diffusion was unreliable for discrete text)",
            "Asking how the AmbientField dual-timescale conscious slow state prevents catastrophic forgetting by physics",
            "Asking about ResonanceDetector, EntanglementBroker, and TopologyShifter parameter migration",
            "Explaining BitNet 1.58b ternary weights {-1, 0, +1} and metabolic efficiency",
            "Simple friendly greeting: hello, how are you today?",
            "Asking for a quick self-introduction as an AI pair-programmer and reasoning assistant",
        ],
        "prompt_template": "Generate a natural, realistic dialogue where the user asks about: {topic}. Ensure the assistant clearly identifies as PCEI and provides an accurate, engaging response.",
    },
    "arithmetic": {
        "name": "Elementary Arithmetic & Direct Mathematics",
        "weight": 0.25,
        "topics": [
            "Basic single-digit and two-digit addition (e.g. 2+2, 7+8, 45+37)",
            "Basic subtraction with positive and negative results (e.g. 15-8, 100-47, 5-12)",
            "Multiplication tables and mental math (e.g. 6*7, 12*11, 25*4, 15*15)",
            "Integer division and remainders (e.g. 100 / 4, 45 / 9, 29 divided by 5)",
            "Fractions addition and simplification (e.g. 1/2 + 1/3, simplifying 18/24)",
            "Percentages and discounts (e.g. 20% of 150, 15% tip on $60)",
            "Simple single-variable linear equations (e.g. 3x + 5 = 20, solve for x)",
            "Elementary geometry formulas (e.g. area of rectangle 6x4, area of circle with radius r=3)",
            "Simple real-world word problems (e.g. sharing 24 apples among 6 friends)",
            "Order of operations (PEMDAS) calculation with parentheses and multiplication",
        ],
        "prompt_template": "Generate a concise mathematical question and answer pair about: {topic}. The question should be direct, and the assistant must state the exact, correct answer clearly with minimal, clean steps.",
    },
    "facts": {
        "name": "Core World Knowledge & Geography",
        "weight": 0.25,
        "topics": [
            "Capital cities of world nations (France/Paris, Japan/Tokyo, Canada/Ottawa, Germany/Berlin, Australia/Canberra)",
            "Continents, oceans, major rivers, and tallest mountains (Everest, Nile, Pacific, Amazon)",
            "Fundamental chemical elements, symbols, and atomic numbers (Hydrogen, Oxygen, Carbon, Gold, Iron)",
            "The solar system: planets in order from the sun, the moon, and basic astronomy",
            "Key milestones in human history (invention of printing press, Moon landing in 1969, World War II end in 1945)",
            "Basic human anatomy and biology (organs, heart circulation, DNA function, red blood cells)",
            "Units of measurement and conversion (meters to kilometers, Celsius to Fahrenheit freezing points, liters to milliliters)",
            "Major authors, literature, and philosophy (Shakespeare, Homer, Plato, Descartes)",
            "Basic computer science terms (CPU, RAM, GPU, operating system, algorithm, bit vs byte)",
            "Earth science: tectonic plates, water cycle, seasons, and atmosphere layers",
        ],
        "prompt_template": "Generate a direct factual question and answer pair about: {topic}. The assistant must provide the accurate, unambiguous answer directly.",
    },
    "commonsense": {
        "name": "Everyday Physics & Commonsense Reasoning",
        "weight": 0.15,
        "topics": [
            "Why ice floats on liquid water (density decrease of water crystal lattice upon freezing)",
            "Why the sky is blue during the day and red at sunset (Rayleigh scattering)",
            "How gravity affects falling objects in vacuum vs air resistance",
            "Why sweating cools the human body (evaporative cooling)",
            "Boiling point and freezing point of water at sea level in Celsius and Fahrenheit",
            "How mirrors reflect light and why reflections appear reversed horizontally",
            "Why heavy metal ships can float while a small metal coin sinks (Archimedes principle & buoyant displacement)",
            "How sound travels through air vs water vs vacuum",
            "Everyday causal reasoning (what happens if you leave ice cream in the sun, why plants need light for photosynthesis)",
            "Simple temporal sequencing (which happens first: planting seed or harvesting fruit)",
        ],
        "prompt_template": "Generate a clear, intuitive question and explanation about everyday physics/common sense: {topic}. The assistant should give a concise, physically accurate explanation understandable to a general audience.",
    },
    "code_logic": {
        "name": "Python Code & Deductive Logic",
        "weight": 0.15,
        "topics": [
            "Write a Python function to find the maximum number in a list without max()",
            "Write a Python function to check if a string is a palindrome",
            "Write a Python function to calculate the factorial of a number iteratively",
            "Write a Python function to reverse a list in-place",
            "Write a Python function to count the frequency of each word in a string",
            "Write a Python function to check if an integer is a prime number",
            "Deductive logic syllogism (All A are B, All B are C, X is A, what is X?)",
            "Simple constraint satisfaction puzzle with 3 people and 3 distinct colors",
            "Binary search algorithm implementation in Python with clean comments",
            "Simple if-else classification function in Python for grading student scores",
        ],
        "prompt_template": "Generate a programming or logic challenge about: {topic}. The assistant should provide a clean, correct, bug-free Python function or step-by-step deductive proof.",
    },
}


# ═══════════════════════════════════════════════════════════════
# 3. ROBUST OLLAMA CLIENT FOR GPT-OSS-120B / 20B
# ═══════════════════════════════════════════════════════════════

class OllamaTeacherClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL, timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def check_connection(self) -> Tuple[bool, str]:
        try:
            r = self.session.get(f"{self.endpoint}/api/tags", timeout=6)
            if r.status_code != 200:
                return False, f"Ollama HTTP status {r.status_code}"
            models = [m.get("name", "") for m in r.json().get("models", [])]
            matched = [m for m in models if self.model in m or m in self.model]
            if not matched:
                cloud_models = [m for m in models if "cloud" in m or "gpt" in m or "120b" in m or "20b" in m]
                return False, f"Model '{self.model}' not found. Available candidates: {cloud_models or models[:5]}"
            return True, f"Connected to Ollama. Model: {matched[0]}"
        except Exception as e:
            return False, f"Connection failed to {self.endpoint}: {e}"

    def generate(self, user_prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> Optional[str]:
        payload = {
            "model": self.model,
            "system": TEACHER_SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.15
            }
        }
        for attempt in range(3):
            try:
                r = self.session.post(f"{self.endpoint}/api/generate", json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    raw = r.json().get("response", "").strip()
                    if raw:
                        return raw
            except requests.exceptions.RequestException as e:
                time.sleep(1.5 * (attempt + 1))
        return None


# ═══════════════════════════════════════════════════════════════
# 4. CURRICULUM PIPELINE & PARSER
# ═══════════════════════════════════════════════════════════════

def clean_dialogue(text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts (prompt, response) from raw teacher output.
    Ensures format:
      prompt = "USER: ...\nASSISTANT:"
      response = " ..."
    """
    if "USER:" not in text or "ASSISTANT:" not in text:
        return None

    cleaned = text.replace("```markdown", "").replace("```text", "").replace("```", "").strip()

    parts = cleaned.split("ASSISTANT:")
    if len(parts) < 2:
        return None

    user_part = parts[0].strip()
    assistant_part = "ASSISTANT:".join(parts[1:]).strip()

    if not user_part.startswith("USER:"):
        idx = user_part.find("USER:")
        if idx != -1:
            user_part = user_part[idx:].strip()

    prompt_formatted = f"{user_part}\nASSISTANT:"
    response_formatted = f" {assistant_part}"

    return prompt_formatted, response_formatted


def sync_to_cloud(local_file: Path) -> bool:
    """Syncs the generated curriculum to the remote RTX 4090 host."""
    if not local_file.exists() or local_file.stat().st_size == 0:
        print("[Sync] Local file empty or missing. Skipping sync.")
        return False

    print(f"\n[Sync] Uploading {local_file.name} -> {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DEST} ...")
    cmd = [
        "rsync", "-avz", "--partial", "--inplace",
        "-e", f"ssh -i {SSH_KEY} -p {REMOTE_PORT} -o StrictHostKeyChecking=no",
        str(local_file),
        f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DEST}"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            print("[Sync] Successfully synced foundational teacher curriculum to remote cloud!")
            return True
        else:
            print(f"[Sync Error] {res.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[Sync Exception] {e}")
        return False


def run_curriculum_generation(
    client: OllamaTeacherClient,
    output_file: Path,
    num_samples: int = 100,
    temperature: float = 0.7,
    sync_on_finish: bool = True
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    domain_keys = list(CURRICULUM_TAXONOMY.keys())
    domain_weights = [CURRICULUM_TAXONOMY[k]["weight"] for k in domain_keys]

    print("═" * 75)
    print(f"  PCEI FOUNDATIONAL TEACHER GENERATOR")
    print(f"  Teacher Model: {client.model} | Target: {output_file.name}")
    print(f"  Target Samples: {num_samples} | Identity: PCEI (846M BitNet 1.58b)")
    print("═" * 75)

    existing_count = 0
    if output_file.exists():
        with open(output_file, "r") as f:
            existing_count = sum(1 for _ in f)
        print(f"[Dataset] Existing samples in output file: {existing_count}")

    stop_requested = False
    def sig_handler(sig, frame):
        nonlocal stop_requested
        print("\n[Signal] Interrupt received. Saving current batch and exiting cleanly...")
        stop_requested = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    successful = 0
    start_time = time.time()

    with open(output_file, "a", encoding="utf-8") as f_out:
        for i in range(num_samples):
            if stop_requested:
                break

            domain_key = random.choices(domain_keys, weights=domain_weights, k=1)[0]
            domain_info = CURRICULUM_TAXONOMY[domain_key]
            topic = random.choice(domain_info["topics"])
            generator_prompt = domain_info["prompt_template"].format(topic=topic)

            t0 = time.time()
            raw_text = client.generate(generator_prompt, temperature=temperature)
            gen_sec = time.time() - t0

            if not raw_text:
                print(f"[{i+1}/{num_samples}] Failed to generate sample for: {topic[:40]}...")
                continue

            parsed = clean_dialogue(raw_text)
            if not parsed:
                prompt_f = f"USER: {topic}\nASSISTANT:"
                response_f = f" {raw_text}"
            else:
                prompt_f, response_f = parsed

            full_text = f"{prompt_f}{response_f}<|endoftext|>"

            record = {
                "prompt": prompt_f,
                "response": response_f,
                "text": full_text,
                "domain": domain_key,
                "topic": topic,
                "model": client.model,
                "source": "pcei_foundation_teacher",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            successful += 1

            total_elapsed = time.time() - start_time
            rate = successful / total_elapsed if total_elapsed > 0 else 0
            sample_preview = response_f.strip().replace("\n", " ")[:65]

            print(
                f"[{i+1:03d}/{num_samples}] [{domain_key:12s}] {topic[:30]:30s} -> "
                f"\"{sample_preview}...\" ({gen_sec:.2f}s | {rate:.1f} samples/s)",
                flush=True
            )

            # Rolling sync to cloud every 50 samples
            if successful > 0 and successful % 50 == 0 and sync_on_finish:
                sync_to_cloud(output_file)

    print("\n" + "═" * 75)
    print(f"  Generation Complete! Successfully generated {successful} high-density samples.")
    print(f"  Total records in {output_file.name}: {existing_count + successful}")
    print("═" * 75)

    if sync_on_finish and successful > 0:
        sync_to_cloud(output_file)


# ═══════════════════════════════════════════════════════════════
# 5. CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PCEI Foundational Teacher Generator")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL}, e.g. gpt-oss:120b-cloud, gpt-oss:20b-cloud)")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT,
                        help=f"Ollama HTTP endpoint (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to generate (default: 100)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--test-query", action="store_true",
                        help="Run a quick test query and print result without saving")
    parser.add_argument("--sync", action="store_true", default=True,
                        help="Sync output file to remote Vast.ai GPU instance after completion")
    args = parser.parse_args()

    client = OllamaTeacherClient(endpoint=args.endpoint, model=args.model)
    ok, msg = client.check_connection()
    if not ok:
        print(f"\n[Warning] {msg}")
        if args.model != "gpt-oss:20b-cloud":
            print("[Fallback] Trying fallback model: gpt-oss:20b-cloud...")
            client.model = "gpt-oss:20b-cloud"
            ok, msg = client.check_connection()
        if not ok:
            print(f"[Fatal] Could not connect to Ollama model. Ensure Ollama is running.\nDetail: {msg}")
            sys.exit(1)

    print(f"[Ollama] {msg}")

    if args.test_query:
        print("\n--- Running Test Query (Identity & Math) ---")
        prompt = "Generate a question asking who the AI is and what 12 * 12 equals."
        res = client.generate(prompt, temperature=args.temperature)
        print("\nRAW OUTPUT:")
        print(res)
        parsed = clean_dialogue(res)
        if parsed:
            p, r = parsed
            print(f"\nPARSED PROMPT:\n{p}")
            print(f"\nPARSED RESPONSE:\n{r}")
        sys.exit(0)

    run_curriculum_generation(
        client=client,
        output_file=Path(args.output),
        num_samples=args.samples,
        temperature=args.temperature,
        sync_on_finish=args.sync
    )


if __name__ == "__main__":
    main()
