import torch

from src.model import BFOR_model

input = torch.randn(size=(2, 3, 448, 448)).to("cuda")
model = BFOR_model(128, 0.2).to("cuda")

print(model(input)["sml"]["w"].shape)
