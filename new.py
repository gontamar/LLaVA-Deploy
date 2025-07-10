import torch
import numpy
print(torch.__version__)
print(numpy.__version__)
print(torch.cuda.is_available())  # Should be True if using CUDA
