from src.datasets import PascalVOC

train_set = PascalVOC(path="./Data")

print(train_set[0])
