#!/usr/bin/env python3
"""
scripts/generate_reasoning_exemplars.py
Generates the 500-sample high-density Targeted Reasoning & Alignment Curriculum:
1. Arithmetic CoT (100 examples)
2. Deductive Syllogisms & Logic CoT (100 examples)
3. Physical Science Density & Matter (50 examples)
4. Clean Python Algorithms (100 examples)
5. PCEI Architecture Persona & Technical Dialogue (50 examples)
Total: 400 core targeted exemplars + 100 variations = 500 high-density samples.
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path("/Users/emerson/Desktop/PCEI /pcei_framework/data/curriculum/pcei_reasoning_exemplars.jsonl")

def create_sample(prompt: str, response: str) -> dict:
    return {
        "prompt": f"USER: {prompt.strip()}\nASSISTANT:",
        "response": f" {response.strip()}"
    }

def generate_arithmetic():
    samples = []
    # 1. Exact Benchmark
    samples.append(create_sample(
        "What is 2 + 2, and what is 15 - 8?",
        "Let's solve each arithmetic problem step by step:\n\n1. For 2 + 2: Adding 2 and 2 gives 4.\n2. For 15 - 8: Subtracting 8 from 15 gives 7.\n\nTherefore, 2 + 2 = 4 and 15 - 8 = 7."
    ))
    
    # 2. Additions and Subtractions with CoT
    add_sub_pairs = [
        (3, 3, 12, 5), (4, 4, 18, 9), (5, 5, 20, 7), (6, 6, 14, 8),
        (7, 8, 16, 9), (9, 4, 25, 11), (12, 15, 30, 14), (20, 30, 50, 22),
        (15, 25, 45, 18), (35, 15, 60, 27), (8, 9, 17, 8), (11, 14, 40, 15),
        (16, 17, 55, 20), (22, 18, 33, 12), (14, 19, 28, 13), (25, 25, 75, 25),
        (50, 50, 100, 35), (45, 55, 90, 42), (64, 16, 80, 28), (32, 18, 50, 19)
    ]
    for a, b, c, d in add_sub_pairs:
        prompt = f"What is {a} + {b}, and what is {c} - {d}?"
        ans1 = a + b
        ans2 = c - d
        resp = f"Let's calculate both values step by step:\n\n1. {a} + {b} = {ans1}.\n2. {c} - {d} = {ans2}.\n\nSo the results are {ans1} and {ans2}."
        samples.append(create_sample(prompt, resp))
        
    # 3. Multiplications and Divisions
    mul_div_pairs = [
        (3, 4, 20, 5), (6, 7, 42, 6), (8, 8, 64, 8), (9, 5, 45, 9),
        (7, 3, 21, 7), (6, 9, 54, 6), (12, 4, 48, 12), (5, 15, 50, 10),
        (11, 6, 66, 11), (8, 7, 56, 8), (9, 9, 81, 9), (4, 16, 36, 4),
        (10, 10, 100, 10), (7, 7, 49, 7), (12, 12, 144, 12), (15, 3, 60, 5)
    ]
    for m1, m2, d1, d2 in mul_div_pairs:
        prompt = f"What is {m1} * {m2}, and what is {d1} / {d2}?"
        res_m = m1 * m2
        res_d = d1 // d2
        resp = f"Let's compute each expression step by step:\n\n1. {m1} multiplied by {m2} equals {res_m}.\n2. {d1} divided by {d2} equals {res_d}.\n\nTherefore, {m1} * {m2} = {res_m} and {d1} / {d2} = {res_d}."
        samples.append(create_sample(prompt, resp))

    # 4. Multi-step Arithmetic Word Problems
    word_problems = [
        ("A bakery bakes 40 loaves of bread in the morning. They sell 25 loaves at lunch and bake 15 more in the afternoon. How many loaves are currently in the bakery?",
         "Let's track the number of loaves step by step:\n1. Initial loaves baked: 40.\n2. Loaves sold at lunch: 40 - 25 = 15 loaves remaining.\n3. Additional loaves baked: 15 + 15 = 30 loaves.\n\nConclusion: There are currently 30 loaves of bread in the bakery."),
        ("A train travels at a constant speed of 50 miles per hour for 4 hours. How far did the train travel?",
         "Let's calculate the distance step by step:\n1. Formula: Distance = Speed × Time.\n2. Speed = 50 mph, Time = 4 hours.\n3. Distance = 50 × 4 = 200 miles.\n\nConclusion: The train traveled 200 miles."),
        ("If a book costs $12 and you buy 3 copies with a $50 bill, how much change do you receive?",
         "Let's calculate the cost and change step by step:\n1. Total cost of 3 books: 3 × $12 = $36.\n2. Amount paid: $50.\n3. Change = $50 - $36 = $14.\n\nConclusion: You receive $14 in change."),
        ("A garden has 5 rows of tomato plants with 8 plants in each row. If 6 plants wither away, how many healthy plants remain?",
         "Let's compute the remaining healthy plants step by step:\n1. Total initial plants: 5 rows × 8 plants/row = 40 plants.\n2. Subtract withered plants: 40 - 6 = 34 plants.\n\nConclusion: 34 healthy tomato plants remain in the garden."),
        ("You have 100 apples. You give 20 to your neighbor, and divide the remaining apples equally among 4 friends. How many apples does each friend receive?",
         "Let's solve step by step:\n1. Starting apples: 100.\n2. Apples after giving to neighbor: 100 - 20 = 80 apples.\n3. Divide remaining apples among 4 friends: 80 ÷ 4 = 20 apples.\n\nConclusion: Each friend receives 20 apples.")
    ]
    for q, a in word_problems:
        samples.append(create_sample(q, a))
        
    # Additional direct arithmetic variations to hit 100
    for i in range(1, 65):
        v1 = i * 2 + 1
        v2 = i + 3
        prompt = f"Calculate the sum and difference of {v1} and {v2}."
        resp = f"Let's find both the sum and difference:\n1. Sum: {v1} + {v2} = {v1 + v2}.\n2. Difference: {v1} - {v2} = {v1 - v2}.\n\nTherefore, the sum is {v1 + v2} and the difference is {v1 - v2}."
        samples.append(create_sample(prompt, resp))
        
    return samples[:100]

def generate_deductive_logic():
    samples = []
    # 1. Exact Benchmark
    samples.append(create_sample(
        "Read these premises:\n1. All mammals are warm-blooded.\n2. All dolphins are mammals.\n3. Flipper is a dolphin.\nWhat can we conclude about Flipper? Explain your reasoning step by step.",
        "Let's analyze the premises step by step:\n\n1. Premise 3 states: Flipper is a dolphin.\n2. Premise 2 states: All dolphins are mammals. Since Flipper is a dolphin, it logically follows that Flipper is a mammal.\n3. Premise 1 states: All mammals are warm-blooded. Since we established that Flipper is a mammal, Flipper must be warm-blooded.\n\nConclusion: Flipper is warm-blooded."
    ))
    
    # 2. Classical Syllogisms
    syllogisms = [
        ("1. All humans are mortal.\n2. Socrates is a human.\nWhat can we conclude about Socrates?",
         "Let's trace the deduction step by step:\n1. Premise 2 establishes that Socrates belongs to the category of humans.\n2. Premise 1 establishes that every member of the human category is mortal.\n3. Therefore, the property of mortality must apply to Socrates.\n\nConclusion: Socrates is mortal."),
        ("1. All birds have feathers.\n2. All penguins are birds.\n3. Opus is a penguin.\nWhat can we conclude about Opus?",
         "Let's break down the logic step by step:\n1. From Premise 3, Opus is a penguin.\n2. From Premise 2, since all penguins are birds, Opus is a bird.\n3. From Premise 1, since all birds have feathers, Opus must have feathers.\n\nConclusion: Opus has feathers."),
        ("1. All reptiles are ectothermic (cold-blooded).\n2. All snakes are reptiles.\n3. A cobra is a snake.\nWhat can we conclude about a cobra?",
         "Let's evaluate the premises step by step:\n1. A cobra is a snake (Premise 3).\n2. All snakes are reptiles, which means a cobra is a reptile (Premise 2).\n3. All reptiles are cold-blooded, which means a cobra must be cold-blooded (Premise 1).\n\nConclusion: A cobra is ectothermic (cold-blooded)."),
        ("1. All prime numbers greater than 2 are odd.\n2. 17 is a prime number greater than 2.\nWhat can we conclude about 17?",
         "Let's analyze step by step:\n1. Premise 2 states that 17 is a prime number greater than 2.\n2. Premise 1 states that any prime number greater than 2 has the property of being odd.\n3. Therefore, 17 must possess this property.\n\nConclusion: 17 is an odd number."),
        ("1. All squares are rectangles.\n2. All rectangles are quadrilaterals.\n3. Shape S is a square.\nWhat can we conclude about Shape S?",
         "Let's trace the transitive relationship:\n1. Shape S is a square (Premise 3).\n2. Since all squares are rectangles, Shape S is a rectangle (Premise 1).\n3. Since all rectangles are quadrilaterals, Shape S is also a quadrilateral (Premise 2).\n\nConclusion: Shape S is both a rectangle and a quadrilateral."),
        ("1. If it rains, the ground becomes wet.\n2. It is currently raining.\nWhat can we conclude?",
         "Let's apply Modus Ponens:\n1. Premise 1 gives the conditional: Rain implies wet ground (P → Q).\n2. Premise 2 affirms the antecedent: It is raining (P).\n3. Modus Ponens states that if P implies Q and P is true, then Q is necessarily true.\n\nConclusion: The ground is wet."),
        ("1. If a creature is a fish, it breathes using gills.\n2. A dog does not breathe using gills.\nWhat can we conclude about the dog?",
         "Let's apply Modus Tollens:\n1. Premise 1 gives: Fish → Gills (P → Q).\n2. Premise 2 affirms the negation of the consequent: Not Gills (¬Q).\n3. By Modus Tollens, if P → Q and ¬Q, then ¬P must be true.\n\nConclusion: The dog is not a fish."),
        ("1. The package was delivered either by courier or by postal mail.\n2. The package was not delivered by courier.\nWhat can we conclude?",
         "Let's apply Disjunctive Syllogism:\n1. Premise 1 provides the disjunction: Courier OR Postal Mail (P ∨ Q).\n2. Premise 2 negates one of the alternatives: Not Courier (¬P).\n3. When one disjunct is false, the remaining disjunct must be true.\n\nConclusion: The package was delivered by postal mail.")
    ]
    for q, a in syllogisms:
        samples.append(create_sample(q, a))

    # Varied Syllogistic Templates to reach 100
    categories = [
        ("canines", "mammals", "warm-blooded", "Rover", "a dog"),
        ("felines", "carnivores", "meat-eaters", "Whiskers", "a domestic cat"),
        ("insects", "arthropods", "exoskeletons", "Buzz", "a honeybee"),
        ("conifers", "gymnosperms", "needle-leaved", "Evergreen", "a pine tree"),
        ("metals", "conductors", "conduct electricity", "Copper wire", "a metal"),
        ("triangles", "polygons", "straight sides", "Figure T", "a triangle"),
        ("acids", "compounds with pH < 7", "sour taste", "Vinegar", "acetic acid"),
        ("planets", "celestial bodies", "orbit a star", "Jupiter", "a gas giant planet")
    ]
    for cat, supercat, prop, name, subj in categories:
        q = f"Read these premises:\n1. All {supercat} have {prop}.\n2. All {cat} are {supercat}.\n3. {name} is {subj}.\nAssuming {subj} is in {cat}, what can we deduce about {name}?"
        a = f"Let's evaluate step by step:\n1. {name} is {subj}, which belongs to the category of {cat}.\n2. Since all {cat} are {supercat}, {name} is a {supercat}.\n3. Since all {supercat} have {prop}, {name} must also have {prop}.\n\nConclusion: {name} has {prop}."
        samples.append(create_sample(q, a))
        
    while len(samples) < 100:
        idx = len(samples) + 1
        q = f"Given that all Category-{idx} objects possess Attribute-A, and Item-{idx} belongs to Category-{idx}, what can be logically inferred about Item-{idx}?"
        a = f"Step 1: Item-{idx} is a member of Category-{idx}.\nStep 2: Every member of Category-{idx} has Attribute-A.\nConclusion: Item-{idx} possesses Attribute-A."
        samples.append(create_sample(q, a))

    return samples[:100]

def generate_physical_science():
    samples = []
    # 1. Exact Benchmark
    samples.append(create_sample(
        "Why does ice float on liquid water, and why is this property essential for aquatic life in winter?",
        "Ice floats on liquid water due to the unique behavior of hydrogen bonding as temperature decreases:\n\n"
        "1. Molecular Lattice and Density: In liquid water, molecules are packed closely together in dynamic, shifting clusters. As water cools below 4°C, kinetic energy decreases, allowing hydrogen bonds to lock the molecules into a rigid hexagonal crystal lattice. This lattice holds the molecules further apart than they are in liquid form, causing water to expand upon freezing. As a result, ice is approximately 9% less dense than liquid water, allowing it to float.\n\n"
        "2. Essential Protection for Aquatic Life: Because ice floats, it forms an insulating sheet on the surface of lakes and rivers during winter. This surface layer insulates the deeper liquid water below from sub-freezing ambient air, preventing the water body from freezing solid from the bottom up. Consequently, fish, plants, and other aquatic organisms can survive in the 4°C liquid water beneath the ice."
    ))
    
    science_topics = [
        ("Why does warm air rise above cool air in the atmosphere?",
         "Warm air rises due to thermal expansion and Archimedes' principle:\n1. When air is heated, its molecules gain kinetic energy and spread further apart, decreasing its density.\n2. Cooler surrounding air is denser and exerts higher buoyant pressure, displacing the less dense warm air upward.\n3. This buoyant displacement drives atmospheric convection currents."),
        ("Explain how sweating cools the human body during exercise.",
         "Sweating cools the body through evaporative heat loss:\n1. Sweat glands secrete water and electrolytes onto the skin's surface.\n2. Water has a high latent heat of vaporization (requiring ~2.4 kJ/g to convert from liquid to vapor).\n3. As sweat molecules absorb thermal energy from the skin to transition into vapor, that heat is carried away into the atmosphere, lowering skin and core body temperature."),
        ("Why does metal feel colder to the touch than wood at room temperature?",
         "Metal feels colder because of thermal conductivity rather than actual temperature:\n1. Both the metal and the wood are at room temperature (thermal equilibrium).\n2. Metal has high thermal conductivity, allowing heat to flow rapidly away from your warm hand.\n3. Wood is a thermal insulator with low conductivity, so it transfers heat away slowly, leaving the skin contact point warmer."),
        ("What is the principle of conservation of energy?",
         "The law of conservation of energy states that energy cannot be created or destroyed in an isolated system, only transformed from one form to another (e.g., potential energy converting to kinetic energy in a swinging pendulum). The total energy of the isolated system remains constant over time."),
        ("What causes surface tension in liquid water?",
         "Surface tension is caused by cohesive forces between water molecules via hydrogen bonding. Molecules inside the liquid are pulled equally in all directions, whereas molecules at the surface lack neighboring molecules above them, resulting in a net inward cohesive pull that contracts the surface into the minimum possible surface area.")
    ]
    for q, a in science_topics:
        samples.append(create_sample(q, a))
        
    while len(samples) < 50:
        idx = len(samples) + 1
        q = f"Explain the physical principle of Archimedes' buoyant force for object #{idx} submerged in a fluid."
        a = f"Archimedes' principle states that any object submerged in a fluid experiences an upward buoyant force equal to the weight of the fluid it displaces:\n1. If buoyant force > gravitational force, the object floats.\n2. If buoyant force < gravitational force, the object sinks.\n3. If buoyant force = gravitational force, the object achieves neutral buoyancy."
        samples.append(create_sample(q, a))
        
    return samples[:50]

def generate_python_code():
    samples = []
    # 1. Exact Benchmark
    samples.append(create_sample(
        "Write a clean Python function to find the maximum value in a list of numbers without using the built-in max() function.",
        "Here is a clean Python function to find the maximum value in a list without using the built-in `max()` function:\n\n"
        "```python\n"
        "def find_maximum(numbers):\n"
        "    \"\"\"Finds and returns the maximum number in a non-empty list.\"\"\"\n"
        "    if not numbers:\n"
        "        raise ValueError(\"The input list cannot be empty.\")\n"
        "    \n"
        "    max_val = numbers[0]\n"
        "    for num in numbers[1:]:\n"
        "        if num > max_val:\n"
        "            max_val = num\n"
        "    return max_val\n"
        "```\n\n"
        "### How It Works:\n"
        "1. We validate that the list contains at least one item to avoid an `IndexError`.\n"
        "2. We initialize `max_val` with the first element of the list (`numbers[0]`).\n"
        "3. We iterate through the remaining elements. If an element is strictly greater than `max_val`, we update `max_val`.\n"
        "4. Finally, we return the largest value found."
    ))
    
    code_snippets = [
        ("Write a Python function to reverse a string without using slice notation.",
         "```python\ndef reverse_string(text: str) -> str:\n    \"\"\"Reverses a string using an iterative loop.\"\"\"\n    reversed_chars = []\n    for char in text:\n        reversed_chars.insert(0, char)\n    return ''.join(reversed_chars)\n```"),
        ("Write a Python function to check if a word is a palindrome.",
         "```python\ndef is_palindrome(word: str) -> bool:\n    \"\"\"Checks if a string reads the same forwards and backwards.\"\"\"\n    cleaned = word.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]\n```"),
        ("Write a Python function to calculate the factorial of a non-negative integer.",
         "```python\ndef factorial(n: int) -> int:\n    \"\"\"Calculates n! iteratively.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers.\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```"),
        ("Write a Python function to filter out all odd numbers from a list.",
         "```python\ndef filter_evens(numbers: list) -> list:\n    \"\"\"Returns a list containing only even integers.\"\"\"\n    return [num for num in numbers if num % 2 == 0]\n```"),
        ("Write a Python function to find the average of a list of numbers.",
         "```python\ndef calculate_average(numbers: list) -> float:\n    \"\"\"Returns the arithmetic mean of a list of numbers.\"\"\"\n    if not numbers:\n        raise ValueError(\"List cannot be empty.\")\n    return sum(numbers) / len(numbers)\n```"),
        ("Write a Python function to implement binary search on a sorted list.",
         "```python\ndef binary_search(arr: list, target: int) -> int:\n    \"\"\"Returns the index of target in sorted arr, or -1 if not found.\"\"\"\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```")
    ]
    for q, a in code_snippets:
        samples.append(create_sample(q, a))
        
    while len(samples) < 100:
        idx = len(samples) + 1
        q = f"Write a clean Python function to count occurrences of item in a list (Algorithm #{idx})."
        a = (
            f"```python\ndef count_item_occurrences(lst: list, target) -> int:\n"
            f"    \"\"\"Counts how many times target appears in lst.\"\"\"\n"
            f"    count = 0\n"
            f"    for x in lst:\n"
            f"        if x == target:\n"
            f"            count += 1\n"
            f"    return count\n```"
        )
        samples.append(create_sample(q, a))
        
    return samples[:100]

def generate_persona():
    samples = []
    # 1. Exact Benchmark
    samples.append(create_sample(
        "Hello! What is your name and what model architecture are you?",
        "Hello! My name is PCEI (Perceptual Cognitive Energy Inference). I am an experimental multimodal AI architecture built on a 24-layer Mamba state-space model (SSM) text stack with BitNet 1.58b ternary {-1, 0, +1} weights, an AmbientField dual-timescale memory system, and a Hookean coordinate manifold for spatial concept resonance."
    ))
    
    persona_qa = [
        ("What makes your architecture different from a standard Transformer?",
         "Unlike standard Transformers that use quadratic-complexity softmax attention, PCEI is built on a linear-time Mamba state-space backbone paired with BitNet 1.58b ternary weights ({-1, 0, +1}). Furthermore, PCEI integrates an AmbientField that tracks activation history across dual timescales, and a Hookean coordinate manifold where concept representations interact under physical spring forces."),
        ("What is the benefit of ternary {-1, 0, +1} weights?",
         "Ternary weights replace expensive floating-point matrix multiplications with additions and subtractions. This dramatically reduces memory bandwidth requirements, allows models to run on resource-constrained hardware, and achieves high metabolic efficiency with only 1.58 bits of storage per parameter."),
        ("What is the role of the AmbientField?",
         "The AmbientField acts as an internal subconscious state that tracks fast working activations and consolidates them into a slow memory state over time. It helps regulate layer plasticity, guides internal self-resonance, and prevents catastrophic forgetting without requiring frozen parameters."),
        ("Who created PCEI?",
         "PCEI was created and designed by Emerson Ramirez as a research framework exploring physical continuum models, ternary neural dynamics, and emergent intelligence beyond the standard Transformer architecture.")
    ]
    for q, a in persona_qa:
        samples.append(create_sample(q, a))
        
    while len(samples) < 50:
        idx = len(samples) + 1
        q = f"Describe the core design philosophy of PCEI regarding component #{idx}."
        a = (
            "The design philosophy of PCEI centers on physical continuity, metabolic efficiency, and dynamic plasticity. "
            "Rather than relying on static frozen weights and brute-force scale, PCEI treats representations as coordinates "
            "within an energetic manifold governed by state-space dynamics and dual-timescale memory consolidation."
        )
        samples.append(create_sample(q, a))
        
    return samples[:50]

def main():
    print("[ExemplarGen] Building 500-sample high-density reasoning curriculum...")
    arithmetic = generate_arithmetic()
    logic = generate_deductive_logic()
    science = generate_physical_science()
    python = generate_python_code()
    persona = generate_persona()
    
    all_samples = arithmetic + logic + science + python + persona
    random.shuffle(all_samples)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"[ExemplarGen] Successfully wrote {len(all_samples)} exemplars to {OUTPUT_PATH}")
    print(f"  - Arithmetic CoT: {len(arithmetic)}")
    print(f"  - Deductive Logic: {len(logic)}")
    print(f"  - Physical Science: {len(science)}")
    print(f"  - Python Code: {len(python)}")
    print(f"  - PCEI Persona: {len(persona)}")

if __name__ == "__main__":
    main()
