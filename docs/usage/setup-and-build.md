# Setup And Build

This project can run its CPU reference path without compiling the custom PagedAttention CUDA kernel. CUDA decode requires a CUDA-capable machine, `nvcc`, and the shared libraries produced by `compile_cuda.sh`.

## Recommended Working Directory

Run commands from the repository root:

```bash
cd 11868-course-project
```

The scripts insert `.` into `sys.path`, so running from another directory can import the wrong `minitorch` package.

## Python Environment

Create and activate a Python environment with Python 3.10 or newer.

Linux, WSL, or PSC-style shell:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## CPU-Only Smoke Check

The CPU path uses MiniTorch `FastOps` and Python reference PagedAttention:

```bash
python project/run_inference.py \
  --backend cpu \
  --decode-backend ref \
  --batch-size 1 \
  --prompt-len 8 \
  --max-new-tokens 4
```

This does not require compiled `paged_attention.so`.

## CUDA Build

CUDA builds use the Bash script `compile_cuda.sh`. On Windows, run it from WSL, a Linux shell, or a CUDA-enabled cluster environment rather than plain PowerShell.

If your environment uses modules:

```bash
module load cuda/12.4
```

Then compile:

```bash
bash compile_cuda.sh
```

Expected outputs:

```text
minitorch/cuda_kernels/combine.so
minitorch/cuda_kernels/softmax_kernel.so
minitorch/cuda_kernels/layernorm_kernel.so
minitorch/cuda_kernels/paged_attention.so
```

## CUDA Smoke Check

After compiling, run:

```bash
python project/run_inference.py \
  --backend cuda \
  --decode-backend cuda \
  --compare-to-ref \
  --batch-size 1 \
  --prompt-len 8 \
  --max-new-tokens 4
```

`--compare-to-ref` runs the Python reference decode beside the CUDA kernel and checks that the outputs are close.

## CUDA Context Caveat

The project intentionally initializes PyTorch CUDA before importing numba CUDA pieces in `minitorch/__init__.py`. This avoids a runtime/driver API ordering issue where later CUDA runtime calls can fail even when a GPU is present.

If CUDA fails in a surprising way, check:

- `paged_attention.so` exists under `minitorch/cuda_kernels/`.
- `compile_cuda.sh` used `--cudart shared` for `paged_attention.cu`.
- The process imports the repository's `minitorch`, not another installed copy.
- A CUDA device is visible in the environment.

## Dependency Notes

Important packages from `requirements.txt`:

- `numpy`: cache storage and numerical reference work.
- `pytest` and `hypothesis`: tests.
- `numba` and `pycuda`: MiniTorch CUDA infrastructure.
- `matplotlib`: benchmark plots.
- `tqdm`: progress helpers used by inherited code paths.

## Clean Rebuild

If CUDA libraries look stale, remove compiled shared objects and rebuild:

```bash
rm -f minitorch/cuda_kernels/*.so
bash compile_cuda.sh
```

Do this from a shell that has `nvcc` on `PATH`.
