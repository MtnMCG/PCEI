from src.pcei import PCEINexus

model = PCEINexus()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
