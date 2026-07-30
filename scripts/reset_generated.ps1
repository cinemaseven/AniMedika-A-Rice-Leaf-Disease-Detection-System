$ErrorActionPreference = "Stop"

Write-Host "This removes generated splits, experiment outputs, and experiment models."
Write-Host "It does NOT remove the six raw class folders under dataset."
$confirmation = Read-Host "Type RESET to continue"

if ($confirmation -ne "RESET") {
    Write-Host "Reset cancelled."
    exit 0
}

Remove-Item -Recurse -Force dataset\split -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force outputs\experiments -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force models\experiments -ErrorAction SilentlyContinue

python src\config.py
Write-Host "Generated folders were reset."
