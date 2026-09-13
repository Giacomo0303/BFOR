import torch

from src.model import BFOR_model

input = torch.randn(size=(1, 3, 448, 448)).to("cuda")
model = BFOR_model().to("cuda")

print(model(input))
