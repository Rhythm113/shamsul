#!/bin/bash
# run.sh - Run script for shamsul (macOS & Linux)
set -euo pipefail

echo "==> Starting shamsul..."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec uv run shamsul
