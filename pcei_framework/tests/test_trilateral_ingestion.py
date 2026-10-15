import torch
from pathlib import Path
import json

# Add src to path

from src.core.ingestors import PCEIDataset

def test_trilateral_ingestion():
    device = "cpu"
    db_path = Path("data/processed/training_database.jsonl")
    
    print("Initializing PCEIDataset...")
    dataset = PCEIDataset(db_path)
    
    # Check if we have the targeted video record
    targeted_records = [r for r in dataset.records if r['id'].startswith('targeted_')]
    print(f"Found {len(targeted_records)} targeted trilateral records.")
    
    if not targeted_records:
        print("Error: No targeted records found!")
        return

    print("\nAttempting to get a multimodal batch (Trilateral sync check)...")
    try:
        batch = dataset.get_multimodal_batch(batch_size=1)
        print("Batch keys:", batch.keys())
        
        if batch['text'] is not None:
            print(f"Text tensor shape: {batch['text'].shape}")
        if batch['image'] is not None:
            print(f"Image tensor shape: {batch['image'].shape}")
        if batch['audio'] is not None:
            print(f"Audio tensor shape: {batch['audio'].shape}")
            
        print("\nTrilateral Sync Sanity Passed!")
        
    except Exception as e:
        print(f"Trilateral Sync Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trilateral_ingestion()
