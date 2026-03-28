# Conda Build Management

Conda/Miniconda is used to manage multiple Python versions across **continuum**, **unified-semantic-compressor** (USC), and related projects.

## Install Miniconda

```powershell
winget install -e --id Anaconda.Miniconda3 --accept-package-agreements --accept-source-agreements
```

Close and reopen your terminal, or run `conda init powershell` and restart.

### First-time setup: Accept Terms of Service

Run once after installing Miniconda:

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

## Environments

| Environment | Python | Purpose |
|-------------|--------|---------|
| `continuum` | 3.12 | Continuum library server, Entropythief ring, serve_library |
| `usc` | 3.12 | USC package, video pipeline (BLIP, CogVideoX), tests |

Python 3.12 is used for USC because PyTorch/diffusers can crash on Python 3.14.

## Setup

### USC (unified-semantic-compressor)

```powershell
cd C:\Users\John\unified-semantic-compressor
conda env create -f conda-environment.yml
conda activate usc
pytest tests/ -v
```

### Continuum

Continuum depends on USC. Two options:

**Option A: USC in same conda env**
```powershell
cd C:\Users\John\continuum
conda env create -f conda-environment.yml
conda activate continuum
pip install -e C:\Users\John\unified-semantic-compressor
pytest tests/ -v
```

**Option B: USC as separate env, PYTHONPATH**
```powershell
conda activate continuum
$env:PYTHONPATH = "C:\Users\John\unified-semantic-compressor"
pytest tests/ -v
```

### Combined env (continuum + USC)

```powershell
cd C:\Users\John\continuum
conda env create -f conda-environment.yml
conda activate continuum
pip install -e C:\Users\John\unified-semantic-compressor
pip install -r requirements.txt
pytest tests/ -v
```

## Quick Reference

```powershell
conda env list                    # List environments
conda activate continuum          # Switch to continuum
conda activate usc                # Switch to USC
conda deactivate                  # Return to base
conda env remove -n continuum     # Remove env
```

## Why Conda?

- **Multiple Python versions**: continuum can stay on 3.14 for non-USC work; USC uses 3.12 for torch.
- **Isolated deps**: torch, diffusers, luigi stay in their own envs.
- **Reproducible**: `conda-environment.yml` pins versions; `conda env export > env.lock.yml` for full lock.
