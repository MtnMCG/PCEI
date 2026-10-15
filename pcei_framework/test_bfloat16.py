import torch
print("bfloat16 attr:", hasattr(torch, 'bfloat16'))
try:
    x = torch.randn(10, dtype=torch.bfloat16, device="mps")
    y = x @ x
    print("bfloat16 matmul success")
except Exception as e:
    print("bfloat16 matmul failed:", e)

try:
    x = torch.randn(10, dtype=torch.float16, device="mps")
    y = x @ x
    print("float16 matmul success")
except Exception as e:
    print("float16 matmul failed:", e)
