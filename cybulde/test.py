import torch
import torch.nn as nn
from torch import optim
import time

#Wanted to see if the local env accepted my gpu

device = torch.device('mps')
nums = 1_000_000

X_cpu = torch.rand(nums,1)
epsilon = torch.normal(0,1,(nums,1))
y_cpu = 3*X_cpu + 4.3 + epsilon

X_gpu = torch.rand(nums,1, device = device)
epsilon_gpu = torch.normal(0,1,(nums,1))
epsilon_gpu = epsilon.to(device)
y_gpu = 3*X_gpu + 4.3 + epsilon_gpu
y_gpu = y_gpu.to(device)

model_cpu = nn.Sequential(
    nn.Linear(1,100),
    nn.ReLU(),
    nn.Linear(100,100),
    nn.ReLU(),
    nn.Linear(100,1)
    )

model_gpu = nn.Sequential(
    nn.Linear(1,100),
    nn.ReLU(),
    nn.Linear(100,100),
    nn.ReLU(),
    nn.Linear(100,1)
    )

model_gpu.to(device)

metric = nn.MSELoss()
optimizer_cpu = optim.SGD(model_cpu.parameters(),lr=.01)
cpu_train_loss = []
cpu_train_time_start = time.time()

for epoch in range(10):
    model_cpu.train()
    optimizer_cpu.zero_grad()
    y_cpu_pred = model_cpu(X_cpu)
    loss = metric(y_cpu_pred, y_cpu)
    loss.backward()
    optimizer_cpu.step()
    cpu_train_loss.append(loss.item())

    print(f"{epoch = } loss = {loss.item()}")

cpu_train_time_end = time.time()

print()
print(60*"#")
print()
print("Starting gpu training loop!")
print()

metric = nn.MSELoss()
optimizer_gpu = optim.SGD(model_gpu.parameters(),lr=.01)
gpu_train_loss = []
gpu_train_time_start = time.time()

for epoch in range(10):
    model_gpu.train()
    optimizer_gpu.zero_grad()
    y_gpu_pred = model_gpu(X_gpu)
    loss = metric(y_gpu_pred, y_gpu)
    loss.backward()
    optimizer_gpu.step()
    gpu_train_loss.append(loss.item())

    print(f"{epoch = } loss = {loss.item()}")

gpu_train_time_end = time.time()

cpu_time = cpu_train_time_end - cpu_train_time_start
gpu_time = gpu_train_time_end - gpu_train_time_start

print()
print(f"Total training time for {nums} inputs on cpu was {cpu_time} seconds")
print(f"Total training time for {nums} inputs on gpu was {gpu_time} seconds")
