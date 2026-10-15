from pathlib import Path
from src.core.ingestors import PCEIDataset

if __name__ == "__main__":
    db_path = Path("data/processed/training_database.jsonl")
    dataset = PCEIDataset(db_path=db_path)
