$ErrorActionPreference = "Stop"

Write-Host "Run this only after all potential near-duplicate pairs have yes/no decisions."
$confirmation = Read-Host "Type REVIEWED to create the group-aware split and run the baseline"

if ($confirmation -ne "REVIEWED") {
    Write-Host "Baseline cancelled."
    exit 0
}

py -3.13 src\environment_check.py
py -3.13 src\config.py
py -3.13 src\audit_dataset.py
py -3.13 src\split_dataset.py
py -3.13 src\dataset.py
py -3.13 src\kfold.py --experiment baseline

Write-Host "Group-aware baseline k-fold run completed."