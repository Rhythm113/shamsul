# setup.ps1 - Setup script for shamsul (Windows)
$ErrorActionPreference = "Stop"

Write-Host "==> Setting up shamsul..."

# 1. Install uv if not present
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing astral uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    # Add uv to PATH for current session
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
} else {
    Write-Host "==> uv is already installed. Updating to latest..."
    try { uv self update } catch { Write-Warning "Could not update uv" }
}

# 2. Install Python 3.14.0
Write-Host "==> Installing Python 3.14.0..."
uv python install 3.14.0

# 3. Synchronize dependencies and virtual environment
Write-Host "==> Synchronizing dependencies..."
uv sync

# 4. Initialize config file if not present
if (-not (Test-Path ".env")) {
    Write-Host "==> Creating .env file from .env.example..."
    Copy-Item ".env.example" ".env"
}

# 5. Initialize config folder ~/.shamsul/ if not present
Write-Host "==> Initializing configuration..."
uv run shamsul-init

Write-Host "==> Setup complete!"
Write-Host "==> You can now run the server using: .\run.ps1"
