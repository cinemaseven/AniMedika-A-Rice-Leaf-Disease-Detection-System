$ErrorActionPreference = "Stop"

Write-Host "This removes generated splits, audit reports, experiment outputs, and experiment models."
Write-Host "It does NOT remove the six raw class folders under dataset."
$confirmation = Read-Host "Type RESET to continue"

if ($confirmation -ne "RESET") {
    Write-Host "Reset cancelled."
    exit 0
}

Remove-Item -Recurse -Force dataset\split -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force outputs\dataset_audit -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force outputs\experiments -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force models\experiments -ErrorAction SilentlyContinue

py -3.13 src\config.py
Write-Host "Generated folders were reset."