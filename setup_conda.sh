#!/bin/bash
set -euo pipefail

echo "=== LLM Training Environment Setup ==="

echo "Installing uv package manager..."
curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | bash
export PATH="$HOME/.local/bin:$PATH"
echo "uv installed successfully at $HOME/.local/bin"
echo "Installing Python 3.12..."
$HOME/.local/bin/uv python install 3.12 || {
  echo "Python 3.12 not available. Installing latest Python 3..."
  $HOME/.local/bin/uv python install 3
}
echo "Creating virtual environment..."
$HOME/.local/bin/uv venv
source .venv/bin/activate

echo "Installing dependencies from requirements.txt..."
$HOME/.local/bin/uv pip install -r requirements.txt

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    echo "Setting up Kaggle credentials..."
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "Kaggle credentials configured."
fi

echo "=== Environment setup complete! ==="
echo "Activate with: source .venv/bin/activate"
echo "Run data download: python engine/00-download-data.py"
echo "Process data: python engine/00-process-data.py"
echo "Train model: python model/01-autoencoder.py"
