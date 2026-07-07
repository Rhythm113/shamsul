# run.ps1 - Run script for shamsul (Windows)
$ErrorActionPreference = "Stop"

Write-Host "==> Starting shamsul..."
$env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
uv run shamsul
