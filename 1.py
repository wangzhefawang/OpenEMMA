
import torch, platform
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
print("python:", platform.python_version())
