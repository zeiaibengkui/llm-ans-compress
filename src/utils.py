import torch


def _detect_device() -> torch.device:
    """优先 XPU（本项目目标设备），其次 CUDA / MPS，最后 CPU。"""
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _detect_device()
