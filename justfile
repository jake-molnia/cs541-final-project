default:
    @just --list

install:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv
    else
        echo "Virtual environment already exists at .venv"
    fi

    echo "Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt

    echo "Setup complete! Activate with: source .venv/bin/activate"

clean:
    rm -rf .venv

run:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ ! -d ".venv" ]; then
        echo "Virtual environment not found. Run 'just install' first."
        exit 1
    fi
