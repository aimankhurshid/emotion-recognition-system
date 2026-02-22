# VS Code Remote Tunnel Setup Guide

## What This Does
Creates a secure tunnel so you can access this Windows laptop from your Mac (or any device) anywhere with internet.

## Step 1: Start the Tunnel on Windows (This Laptop)

### Option A: Using VS Code UI (Easiest)
1. Press `F1` or `Ctrl+Shift+P` to open Command Palette
2. Type: `Remote-Tunnels: Turn on Remote Tunnel Access...`
3. Select it and follow the prompts
4. Sign in with your GitHub or Microsoft account
5. Give your machine a name (e.g., "naman-windows-laptop")
6. VS Code will generate a tunnel URL

### Option B: Using PowerShell (if CLI path is available)
Run this in PowerShell as Administrator:
```powershell
# Navigate to VS Code installation
cd "C:\Users\Naman\AppData\Local\Programs\Microsoft VS Code\bin"

# Start tunnel (will prompt for GitHub/Microsoft login)
.\code.cmd tunnel --accept-server-license-terms
```

### Option C: Download VS Code CLI separately
If the above doesn't work, download the standalone CLI:
1. Go to: https://code.visualstudio.com/docs/remote/tunnels
2. Download the VS Code CLI for Windows
3. Run: `code tunnel --accept-server-license-terms`

## Step 2: Connect from Your Mac

### Method 1: Through vscode.dev (Browser)
1. Go to: https://vscode.dev/tunnel
2. Sign in with the same GitHub/Microsoft account
3. Select your Windows machine from the list
4. You're now connected!

### Method 2: Through VS Code Desktop on Mac
1. Open VS Code on your Mac
2. Press `Cmd+Shift+P` 
3. Type: `Remote-Tunnels: Connect to Tunnel...`
4. Sign in and select your Windows machine
5. Connected!

## Step 3: Keep the Tunnel Running (Windows)

### Option A: Keep VS Code Open
Just leave VS Code running on Windows

### Option B: Run as Background Service
Install tunnel as a Windows service so it runs even when VS Code is closed:
```powershell
# Run as Administrator
code tunnel service install
code tunnel service start
```

## Security Notes
- Tunnel uses Microsoft/GitHub authentication
- All traffic is encrypted
- No port forwarding or firewall configuration needed
- You control access through your account

## Troubleshooting

### If tunnel command not found:
Add VS Code to PATH or use full path:
```powershell
$env:Path += ";C:\Users\Naman\AppData\Local\Programs\Microsoft VS Code\bin"
```

### To check tunnel status:
```powershell
code tunnel status
```

### To stop tunnel:
```powershell
code tunnel kill
# or
code tunnel service uninstall  # if running as service
```

## Quick Start Command
Run this now to start the tunnel:
```powershell
# Try to find code in common locations
$codePaths = @(
    "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd",
    "$env:ProgramFiles\Microsoft VS Code\bin\code.cmd",
    "$env:ProgramFiles(x86)\Microsoft VS Code\bin\code.cmd"
)

$codeCmd = $codePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($codeCmd) {
    Write-Host "Found VS Code at: $codeCmd"
    & $codeCmd tunnel --accept-server-license-terms --name "naman-windows-laptop"
} else {
    Write-Host "VS Code CLI not found in standard locations."
    Write-Host "Please use the UI method (F1 > Remote-Tunnels: Turn on Remote Tunnel Access)"
}
```
