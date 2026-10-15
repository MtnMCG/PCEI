import torch

torch.manual_seed(42)
B, L, D1, D2 = 2, 64, 4, 4
chunk_size = 16

dA = torch.rand(B, L, D1, D2) * 0.9 + 0.1 # bounded away from 0
dB = torch.randn(B, L, D1, D2)

# Loop version
h_loop = torch.zeros_like(dB)
curr = torch.zeros_like(dB[:, 0])
for t in range(L):
    curr = dA[:, t] * curr + dB[:, t]
    h_loop[:, t] = curr

# Chunk version
h_chunk = torch.zeros_like(dB)
curr = torch.zeros_like(dB[:, 0])

for i in range(0, L, chunk_size):
    c_dA = dA[:, i:i+chunk_size]
    c_dB = dB[:, i:i+chunk_size]
    
    log_dA = torch.log(c_dA.clamp(min=1e-8))
    cum_log_dA = torch.cumsum(log_dA, dim=1)
    cum_dA = torch.exp(cum_log_dA)
    
    weighted_dB = c_dB / cum_dA.clamp(min=1e-8)
    cum_weighted_dB = torch.cumsum(weighted_dB, dim=1)
    local_h = cum_dA * cum_weighted_dB
    
    carry_h = curr.unsqueeze(1) * cum_dA
    c_h = local_h + carry_h
    h_chunk[:, i:i+chunk_size] = c_h
    curr = c_h[:, -1]

print("Max diff:", torch.max(torch.abs(h_loop - h_chunk)).item())
