import torch


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

x = torch.ones(2, 2, requires_grad=True)
y = x + 2

z = y * y * 3
out = z.mean()

print(out)