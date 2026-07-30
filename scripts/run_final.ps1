param(
    [Parameter(Mandatory = $true)]
    [string]$Experiment
)

$ErrorActionPreference = "Stop"

Write-Host "Selected experiment: $Experiment"
Write-Host "Only continue after the experiment was selected using k-fold validation."
$confirmation = Read-Host "Type LOCKED to train and evaluate the final model"

if ($confirmation -ne "LOCKED") {
    Write-Host "Final run cancelled."
    exit 0
}

python src\train.py --experiment $Experiment
python src\calibration.py --experiment $Experiment --source validation
python src\evaluate.py --experiment $Experiment --confirm-final
python src\compare_splits.py --experiment $Experiment --confirm-final

Write-Host "Final training, calibration, evaluation, and split comparison completed."
