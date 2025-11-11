# .local_env.ps1
# Load environment variables from .env and set up project environment

# Resolve project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Add src folder to PYTHONPATH
$SrcPath = Join-Path $ProjectRoot "src"
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", $SrcPath, "Process")

Write-Host "Environment loaded for PCCEntropy"