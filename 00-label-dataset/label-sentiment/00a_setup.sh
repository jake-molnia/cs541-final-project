#!/bin/bash
set -e

export UV_LINK_MODE=copy
module load cudnn8.9-cuda12.3/8.9.7.29

echo "=== Installing uv package manager ==="
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv --version

echo "=== Creating Python virtual environment ==="
uv venv
source .venv/bin/activate

echo "=== Installing dependencies with uv ==="
uv pip install -r requirements.txt

echo "=== Installation complete ==="
echo "To activate the virtual environment, run: source .venv/bin/activate"
