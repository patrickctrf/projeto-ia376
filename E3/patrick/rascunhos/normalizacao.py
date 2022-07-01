import torch
from torch import nn

bn = nn.BatchNorm1d(100, affine=False).train()

for i in range(22222):
    bn(torch.randn(20, 100) * 20 + 7)

bn.eval()
uns_eval = bn(torch.ones(20, 100))

bn.train()
uns_treino = bn(torch.ones(20, 100))

print(uns_treino)
