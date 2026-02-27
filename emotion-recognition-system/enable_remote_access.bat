@echo off
echo ================================================================================
echo ENABLE REMOTE ACCESS - WINDOWS RTX
echo ================================================================================
echo.

echo This will enable VSCode Remote Tunnels so you can:
echo   - Work from your Mac or any device
echo   - Access multiple terminals remotely
echo   - Start/stop training from anywhere
echo.

echo ================================================================================
echo METHOD 1: AUTOMATIC (if code command works)
echo ================================================================================
echo.

where code >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo VSCode found in PATH. Starting tunnel...
    code tunnel --accept-server-license-terms
) else (
    echo VSCode 'code' command not found in PATH.
    echo.
    echo ============================================================================
    echo METHOD 2: MANUAL (Use VSCode UI) - RECOMMENDED
    echo ============================================================================
    echo.
    echo Follow these steps:
    echo.
    echo 1. Open VSCode on this Windows machine
    echo 2. Press Ctrl+Shift+P
    echo 3. Type: "Remote Tunnels: Turn on Remote Tunnel Access"
    echo 4. Press Enter
    echo 5. Sign in with GitHub when prompted
    echo 6. Keep VSCode running minimized
    echo.
    echo Once enabled, you can access from anywhere:
    echo   - Browser: https://vscode.dev
    echo   - VSCode Desktop: Install "Remote - Tunnels" extension
    echo.
    echo ============================================================================
    echo.
    pause
)
