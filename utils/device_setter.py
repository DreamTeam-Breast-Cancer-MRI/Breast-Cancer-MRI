import torch


def device_setter():
    return "cuda" if torch.cuda.is_available() else "cpu"
