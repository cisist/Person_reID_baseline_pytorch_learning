import torch
import torchvision
import torchaudio

# 查看版本
print("PyTorch 版本:", torch.__version__)
print("Torchvision 版本:", torchvision.__version__)
print("Torchaudio 版本:", torchaudio.__version__)

# 验证 M4 GPU 加速（MPS 后端）
print("MPS 可用:", torch.backends.mps.is_available())
print("MPS 构建:", torch.backends.mps.is_built())
