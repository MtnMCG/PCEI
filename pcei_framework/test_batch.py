import torch
import warnings
from scripts.train import PCEITokenizer, MAX_SEQ_LEN
from src.pcei import PCEINexus
from src.core.ingestors import PCEIDataset
from pathlib import Path

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

nexus = PCEINexus(vocab_size=50257, device=device)
nexus.train()
optimizer = torch.optim.AdamW(nexus.parameters(), lr=3e-4)

tokenizer = PCEITokenizer()
db_path = Path("data/processed/training_database.jsonl")
dataset = PCEIDataset(db_path, tokenizer=tokenizer)

# Let's get a mean confidence helper for the mock script
def debug_mean_confidence(core_module):
    from src.modules.plasticity import PlasticLayer
    conf_vals = []
    for _, module in core_module.named_modules():
        if isinstance(module, PlasticLayer):
            conf_vals.append(module.slow_field_confidence.item())
    return sum(conf_vals) / max(len(conf_vals), 1)

print("\n[Debug] Running 50 Continuous Batches...")
for step in range(50):
    batch = dataset.get_multimodal_batch(4)
    
    for k in batch:
        if batch[k] is not None:
            batch[k] = batch[k].to(device)

    logits, _, _, _ = nexus.core(
        text_input=batch['text'],
        img_input=batch['image'],
        audio_input=batch['audio'],
        diffusion_timesteps=3,
        slow_state=nexus.ambient_field.slow_state,
        mean_confidence=debug_mean_confidence(nexus.core)
    )

    if logits is not None and batch['text'] is not None:
        targets = batch['text'][:, 1:].contiguous()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(
            logits[:, :targets.size(1), :].reshape(-1, 50257),
            targets.reshape(-1)
        )
        loss.backward()
        print(f"  [{step:02d}] Success! Loss: {loss.item():.4f}")
