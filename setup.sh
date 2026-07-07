#!/bin/bash
# setup.sh - Setup script for shamsul (macOS & Linux)
set -euo pipefail

echo "==> Setting up shamsul..."

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "==> Installing astral uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "==> uv is already installed. Updating to latest..."
    uv self update || true
fi

# 2. Install Python 3.14.0
echo "==> Installing Python 3.14.0..."
uv python install 3.14.0

# 3. Synchronize dependencies and virtual environment
echo "==> Synchronizing dependencies..."
uv sync

# 4. Initialize config file if not present
if [ ! -f .env ]; then
    echo "==> Creating .env file from .env.example..."
    cp .env.example .env
fi

# 5. Initialize config folder ~/.shamsul/ if not present
echo "==> Initializing configuration..."
uv run shamsul-init

echo "==> Setup complete!"
echo "==> You can now run the server using: ./run.sh"
