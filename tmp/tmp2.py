import torch

# 查看是否支持 MPS
print("MPS 是否可用：", torch.backends.mps.is_available())

# 测试速度对比
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)

# CPU 计算
import time
start = time.time()
c = a @ b
print("CPU 耗时：", time.time() - start)

# MPS 计算
a = a.to("mps")
b = b.to("mps")
start = time.time()
c = a @ b
print("MPS 耗时：", time.time() - start)
