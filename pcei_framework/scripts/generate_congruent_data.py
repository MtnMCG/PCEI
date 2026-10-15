import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Generate specific conceptual texts that match the user's uploaded multimodal dataset
custom_pairs = [
    # Death Grips (Audio/Images)
    "Death Grips is an American experimental hip hop group formed in Sacramento, California.",
    "The album Exmilitary features an Aboriginal man from Australia on its cover.",
    "MC Ride is the intense frontman of the experimental hip hop group Death Grips.",
    "The song Guillotine by Death Grips has the iconic lyric 'It goes, it goes, it goes... Guillotine!'",
    "Exmilitary is a 2011 mixtape known for its raw, industrial, and chaotic energy.",
    
    # Nature / Scenes (Images)
    "Emerald Bay in South Lake Tahoe, California is known for its beautiful clear water and pine trees.",
    "A waterfall cutting through granite rockfaces in a dense forest.",
    "A bridge constructed over a rushing river in the woods.",
    "A monochromatic black and white photograph of a serene lake.",
    "Pine trees lining the side of a road as seen through a car windshield.",
    
    # Objects & Animals (Images)
    "A Caucasian human eating a bagel sandwich wrapped in tinfoil inside a Mercedes car.",
    "A shiny red apple resting on a surface.",
    "A capybara resting by the river.",
    "A capybara swimming with its cubs.",
    "A large rodent known as a capybara standing in the grass.",
    
    # Acoustic Guitar (Video)
    "The sound of an acoustic guitar being strummed rhythmically.",
    "Fingers strumming the strings of a wooden acoustic guitar to play chords.",
    
    # Speech (Audio)
    "Steve Jobs gave a highly motivational speech discussing three lessons he learned in his life.",
    "The philosophy of determinism and how it relates to human conversation and free will.",
    "Plato's Republic is a foundational text discussing abstract philosophical concepts."
]

def generate_congruent_data():
    output_path = DATA_DIR / "congruent_concepts.jsonl"
    records = []
    
    # Generate 500 copies of these concepts so there's enough volume to form strong bonds
    for _ in range(500):
        for text in custom_pairs:
            records.append({
                "text": text,
                "type": "congruent_concept",
                "tag": "custom"
            })
            
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(records)} highly congruent synthetic text records.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_congruent_data()
