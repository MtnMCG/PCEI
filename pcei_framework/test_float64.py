import torch
device = "mps"
x = torch.randn(10, dtype=torch.float64).to(device)
y = torch.cumsum(x, dim=0)
z = torch.exp(x)
print(y.dtype, z.dtype)
print("SUCCESS")
