$ErrorActionPreference = "Stop"

Write-Host "Run this only after manually reviewing the duplicate reports."
$confirmation = Read-Host "Type REVIEWED to create the split and run the baseline"

if ($confirmation -ne "REVIEWED") {
    Write-Host "Baseline cancelled."
    exit 0
}

python src\environment_check.py
python src\config.py
python src\split_dataset.py
python src\dataset.py
python src\kfold.py --experiment baseline

Write-Host "Baseline k-fold run completed."
