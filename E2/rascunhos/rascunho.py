import torch
from matplotlib import pyplot as plt


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, new_lr=0.01):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        return


model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=100)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.1)

lrs = []

for i in range(100):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

    if i == 45: set_lr(optimizer, new_lr=10)

plt.plot(range(100), lrs)
plt.show()
