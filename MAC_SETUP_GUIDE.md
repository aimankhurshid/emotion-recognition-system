# Cross-Platform Workspace Setup (Windows RTX ↔ Mac)

## Option 1: Git Sync Workflow (RECOMMENDED)

### On Mac - Initial Setup
```bash
# 1. Clone the repository
cd ~/Documents  # or your preferred location
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion-recognition-system

# 2. Open in VSCode
code .

# 3. Install Python dependencies
pip3 install -r requirements.txt
```

### Daily Workflow

#### On Windows RTX (Training):
```powershell
# Before starting work - pull latest from Mac
git pull origin main

# After training/making changes
git add .
git commit -m "Your message [Windows-RTX]"
git push origin main
```

#### On Mac (Research):
```bash
# Before starting work - pull latest from Windows
git pull origin main

# After research/documentation
git add .
git commit -m "Your message [Mac-Research]"
git push origin main
```

---

## Option 2: VSCode Remote - SSH (Work on Windows from Mac)

### Setup on Windows RTX

#### 1. Install OpenSSH Server
```powershell
# Check if installed
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

# Install if needed
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start service
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Get your IP address
ipconfig | Select-String "IPv4"
```

#### 2. Configure Firewall
```powershell
# Allow SSH through firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

#### 3. Get your Windows username
```powershell
$env:USERNAME
```

### Setup on Mac

#### 1. Install VSCode Extensions
- Remote - SSH (ms-vscode-remote.remote-ssh)
- Remote Explorer

#### 2. Configure SSH Connection
```bash
# Edit SSH config
nano ~/.ssh/config
```

Add this configuration:
```
Host windows-rtx
    HostName YOUR_WINDOWS_IP_ADDRESS
    User YOUR_WINDOWS_USERNAME
    Port 22
```

#### 3. Connect from VSCode
1. Press `Cmd+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Select "windows-rtx"
4. Open folder: `C:\Users\Naman\emotion-recognition-system`

Now you can edit files on Windows directly from your Mac!

---

## Option 3: VSCode Remote Tunnels (Access from Anywhere)

### On Windows RTX

#### 1. Enable Remote Tunnels
```powershell
# Download VSCode CLI if not installed
# Open VSCode Command Palette (Ctrl+Shift+P)
# Type: "Remote Tunnels: Turn on Remote Tunnel Access..."
```

Or use command line:
```powershell
# In your project directory
code tunnel --accept-server-license-terms
```

#### 2. Authenticate with GitHub
- Browser will open
- Sign in with your GitHub account
- Grant permissions

#### 3. Note the tunnel name
- You'll get a URL like: `https://vscode.dev/tunnel/YOUR-MACHINE-NAME/folder`

### On Mac

#### 1. Access via Browser
- Go to: https://vscode.dev
- Sign in with GitHub
- Click on Remote Explorer → Tunnels
- Select your Windows RTX machine

#### 2. Or use VSCode Desktop
```bash
# Install Remote - Tunnels extension
# Cmd+Shift+P → "Remote Tunnels: Connect to Tunnel"
# Select your Windows RTX machine
```

---

## Option 4: Cloud Storage Sync (NOT RECOMMENDED for code)

❌ **Not recommended** because:
- Can cause merge conflicts
- Data folder is huge
- Model files are gigabytes

---

## Recommended Setup: Git + Remote-SSH

### Best of Both Worlds

**For Training (Windows RTX):**
- Work locally on Windows
- Heavy training tasks run on powerful GPU
- Commit and push results

**For Research/Documentation (Mac):**
- Option A: Clone locally, pull/push via Git
- Option B: SSH into Windows and work remotely
- Light tasks like documentation, analysis

---

## VSCode Workspace Configuration

### Save Multi-Root Workspace

On **Windows**, create this file:
**`emotion_recognition_system.code-workspace`**

```json
{
    "folders": [
        {
            "path": "."
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "python",
        "python.terminal.activateEnvironment": true,
        "files.exclude": {
            "**/__pycache__": true,
            "**/*.pyc": true,
            "**/.git": false,
            "data/train": true,
            "data/val": true,
            "results/experiments": false
        },
        "terminal.integrated.cwd": "${workspaceFolder}",
        "git.autofetch": true,
        "git.confirmSync": false
    },
    "extensions": {
        "recommendations": [
            "ms-python.python",
            "ms-python.vscode-pylance",
            "ms-toolsai.jupyter",
            "GitHub.copilot",
            "eamodio.gitlens",
            "ms-vscode-remote.remote-ssh"
        ]
    }
}
```

On **Mac**, same workspace file works!

---

## Quick Start Guide

### Initial Setup (One-time)

#### On Windows RTX:
```powershell
# Already done - repository is at C:\Users\Naman\emotion-recognition-system
git config user.name "aimankhurshid"
git config user.email "your_email@example.com"
```

#### On Mac:
```bash
# Clone repository
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion-recognition-system

# Configure Git
git config user.name "aimankhurshid"
git config user.email "your_email@example.com"

# Install dependencies
pip3 install -r requirements.txt

# Install VSCode Remote-SSH extension (if using SSH)
code --install-extension ms-vscode-remote.remote-ssh
```

---

## File Sync Strategy

### Files to Sync (via Git):
✅ Source code (`.py` files)
✅ Documentation (`.md` files)
✅ Configuration files
✅ Training scripts
✅ Small results (JSON, logs)

### Files NOT to Sync (add to .gitignore):
❌ Large datasets (`data/train/`, `data/val/`)
❌ Model checkpoints (`.pth` files > 100MB)
❌ TensorBoard logs (large event files)
❌ Virtual environments

### Update .gitignore:
I'll add proper ignore rules for large files.

---

## Monitoring Training from Mac

### Option 1: TensorBoard Remote Access (SSH Tunnel)
```bash
# On Mac, create SSH tunnel to Windows
ssh -L 6006:localhost:6006 windows-rtx

# On Windows, TensorBoard is running on port 6006
# Now open on Mac: http://localhost:6006
```

### Option 2: Save Plots/Results to Git
- Visualizations saved as PNG
- Training summaries as JSON
- Push to Git after each epoch/run

---

## Recommended Workflow

### Phase 1: Setup (Do Once)
1. ✅ Windows RTX: Already set up
2. 🔄 Mac: Clone repository
3. 🔄 Mac: Install dependencies
4. 🔄 Optional: Set up Remote-SSH

### Phase 2: Daily Work

#### Training Day (Windows RTX):
```powershell
# Morning: Pull latest
git pull origin main

# Train model (running now!)
python train_anti_overfitting.py

# Evening: Push results
git add results/ train*.py
git commit -m "Training results: epoch 50 completed [Windows-RTX]"
git push origin main
```

#### Research Day (Mac):
```bash
# Morning: Pull latest
git pull origin main

# Analyze results
python visualize_experiment.py --compare

# Update documentation
# Edit markdown files, add findings

# Evening: Push research
git add docs/ *.md notebooks/
git commit -m "Add research findings and analysis [Mac-Research]"
git push origin main
```

---

## Next Steps

Choose your preferred method:

**🎯 Quick Start (Git Only)**
```bash
# On Mac terminal:
git clone https://github.com/aimankhurshid/emotion-recognition-system.git
cd emotion-recognition-system
code .
```

**🚀 Advanced (Remote SSH)**
```bash
# Follow "Option 2: VSCode Remote - SSH" instructions above
```

**☁️ Cloud (Remote Tunnels)**
```powershell
# On Windows: code tunnel
# On Mac: Open https://vscode.dev
```

## What Would You Like?
Let me know which option you prefer and I'll help you set it up step by step!
