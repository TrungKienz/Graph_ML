import platform
import subprocess


def section(title):
    print(f"\n=== {title} ===")


section("System")
print(platform.platform())
print("Python:", platform.python_version())

section("nvidia-smi")
try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print(out.stdout if out.returncode == 0 else out.stderr)
except FileNotFoundError:
    print("nvidia-smi not found on PATH")

section("PyTorch CUDA check")
try:
    import torch

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("device name:", torch.cuda.get_device_name(0))

        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        c = a @ b
        torch.cuda.synchronize()
        print("matmul result shape:", c.shape, "sum:", c.sum().item())
        print("GPU matmul OK")
    else:
        print("No CUDA device visible to PyTorch")
except ImportError:
    print("PyTorch not installed")
