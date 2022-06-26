import torch
from torch import nn


class DummyGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dummy_tensor = nn.Parameter(torch.rand((1, 2, 1024, 128), requires_grad=True))

    def forward(self):
        return self.dummy_tensor


model = DummyGenerator()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

y = torch.randn(1, 2, 1024, 128)

for epoch in range(10000):
    optimizer.zero_grad()
    out = model()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))