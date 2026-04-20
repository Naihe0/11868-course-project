import minitorch, numpy, torch, numba, pytest
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("numba", numba.__version__)
print("numpy", numpy.__version__)
print("minitorch", minitorch.__file__)
