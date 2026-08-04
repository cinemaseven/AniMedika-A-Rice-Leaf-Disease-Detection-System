param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment
)

$ErrorActionPreference = "Stop"

Write-Host "Selected experiment: $Experiment"
Write-Host "Only continue after the experiment was selected using group-aware k-fold validation."
$confirmation = Read-Host "Type LOCKED to train and evaluate the final model"

if ($confirmation -ne "LOCKED") {
    Write-Host "Final run cancelled."
    exit 0
}

py -3.13 src\train.py --experiment $Experiment
py -3.13 src\calibration.py --experiment $Experiment --source validation
py -3.13 src\evaluate.py --experiment $Experiment --confirm-final
py -3.13 src\compare_splits.py --experiment $Experiment --confirm-final

Write-Host "Final training, calibration, evaluation, and split comparison completed."