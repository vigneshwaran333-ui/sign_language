import torch

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)

model = ckpt["model"]

print(model.names)