# Push to GitHub with Windows RTX indicator
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "PUSHING TO GITHUB FROM WINDOWS RTX" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

# Stage all new and modified files
Write-Host "`nStaging files..." -ForegroundColor Yellow
git add train_anti_overfitting.py
git add visualize_experiment.py
git add visualize_training.py
git add TRAINING_STRATEGY.md
git add training/train.py
git add results/training_analysis_complete.png

# Check status
Write-Host "`nGit Status:" -ForegroundColor Yellow
git status

# Commit with Windows RTX indicator
$commitMessage = "Add anti-overfitting training system with comprehensive logging [Windows-RTX]"
Write-Host "`nCommitting with message:" -ForegroundColor Yellow
Write-Host $commitMessage -ForegroundColor White
git commit -m $commitMessage

# Push to GitHub
Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "PUSH COMPLETED" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
