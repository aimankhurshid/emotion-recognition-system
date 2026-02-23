@echo off
echo ================================================================================
echo PUSHING TO GITHUB FROM WINDOWS RTX
echo ================================================================================
echo.

echo Staging files...
git add .

echo.
echo Git Status:
git status -s

echo.
echo Committing changes...
git commit -m "Add anti-overfitting training system with comprehensive logging [Windows-RTX]"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ================================================================================
echo PUSH COMPLETED
echo ================================================================================
pause
