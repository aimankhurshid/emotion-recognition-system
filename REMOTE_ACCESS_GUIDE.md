# Remote Work Setup - Access Windows RTX From Anywhere

## 🚀 VSCode Remote Tunnels (RECOMMENDED)

Access your Windows RTX machine from **anywhere** - Mac, iPad, another PC, or even a web browser!

### Setup on Windows RTX (Do Once)

#### Method 1: Using VSCode UI (Easiest)

1. **Open VSCode on Windows**
2. **Enable Remote Tunnel**:
   - Press `Ctrl+Shift+P`
   - Type: `Remote Tunnels: Turn on Remote Tunnel Access...`
   - Click it

3. **Sign in with GitHub**:
   - Browser will open
   - Sign in with your GitHub account (aimankhurshid)
   - Grant permissions

4. **Note your tunnel name**:
   - VSCode will show: "Tunnel 'YOUR-MACHINE-NAME' is running"
   - Your machine name is likely: `Naman` or `DESKTOP-...`

5. **Keep VSCode running** (minimize it, don't close)

#### Method 2: Using PowerShell

```powershell
# Navigate to VSCode installation
cd "C:\Program Files\Microsoft VS Code\bin"

# Start tunnel
.\code.cmd tunnel --accept-server-license-terms

# Or if VSCode is in PATH:
code tunnel --accept-server-license-terms
```

---

## 🌍 Access From Anywhere

### Option A: Web Browser (Works Everywhere!)

1. Go to: **https://vscode.dev**
2. Click the hamburger menu (☰) in top-left
3. Click **"Remote Explorer"**
4. Select **"Tunnels"**
5. Sign in with GitHub
6. Click on your Windows RTX machine
7. Open folder: `C:\Users\Naman\emotion-recognition-system`

**Now you can:**
- ✅ Edit code live
- ✅ Open multiple terminals
- ✅ Start/stop training
- ✅ Monitor progress
- ✅ Access from iPad, Mac, any device!

### Option B: VSCode Desktop (Mac/Other PC)

1. **Install VSCode** on your Mac/other device
2. **Install Remote Tunnels Extension**:
   - Open VSCode
   - Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows)
   - Search: "Remote - Tunnels"
   - Install it

3. **Connect**:
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P`
   - Type: `Remote Tunnels: Connect to Tunnel`
   - Sign in with GitHub
   - Select your Windows RTX machine
   - Open: `C:\Users\Naman\emotion-recognition-system`

---

## 🖥️ Using Multiple Terminals

Once connected remotely:

### Terminal 1: Training
```powershell
# Start training
python train_anti_overfitting.py
```

### Terminal 2: Monitoring
```powershell
# Watch GPU usage
nvidia-smi -l 1

# Or use Python
watch -n 1 nvidia-smi
```

### Terminal 3: TensorBoard
```powershell
# Start TensorBoard
tensorboard --logdir results/experiments/anti_overfitting_20260223_151402/logs --port 6006
```

### Terminal 4: Git/Development
```powershell
# For git commands, file management, etc.
git status
ls results/
```

### How to Open Multiple Terminals in VSCode:
1. Press `` Ctrl+` `` to open terminal
2. Click the **+** button to add more terminals
3. Or press `Ctrl+Shift+5` to split terminals

---

## ⚡ Quick Start Guide

### On Windows RTX (Setup - Do Once)

```powershell
# Option 1: Through VSCode UI
# Ctrl+Shift+P → "Remote Tunnels: Turn on Remote Tunnel Access..."

# Option 2: Through PowerShell
# Find VSCode installation
$vscodePath = (Get-Command code -ErrorAction SilentlyContinue).Source
if ($vscodePath) {
    code tunnel --accept-server-license-terms
} else {
    # Add VSCode to PATH
    $env:Path += ";C:\Program Files\Microsoft VS Code\bin"
    code tunnel --accept-server-license-terms
}
```

### From Any Device

**Browser**: https://vscode.dev → Sign in → Remote Explorer → Tunnels → Select your machine

**VSCode App**: `Cmd/Ctrl+Shift+P` → "Remote Tunnels: Connect to Tunnel" → Select machine

---

## 🎯 Starting Training Remotely

### From Web Browser (vscode.dev)

1. Connect to your Windows RTX tunnel
2. Open terminal (`` Ctrl+` ``)
3. Run:
```powershell
python train_anti_overfitting.py
```

### Multiple Training Sessions

**Terminal 1**: Main Training
```powershell
python train_anti_overfitting.py
```

**Terminal 2**: Monitor GPU
```powershell
nvidia-smi -l 2
```

**Terminal 3**: TensorBoard
```powershell
tensorboard --logdir results/experiments/ --port 6006
```

**Terminal 4**: View Logs
```powershell
Get-Content results/experiments/anti_overfitting_*/logs/*_training.log -Wait
```

---

## 🔒 Security & Access

### Who Can Access?
- ✅ Only you (authenticated via your GitHub account)
- ✅ Encrypted connection
- ✅ No open ports needed

### Keep Tunnel Running
- Windows must be **logged in** and **VSCode running**
- Minimize VSCode, don't close it
- Or run as service (see below)

### Run Tunnel as Service (Always On)

```powershell
# Install as Windows service
code tunnel service install

# Start service
code tunnel service start

# Check status
code tunnel service status
```

---

## 🎮 Usage Examples

### Example 1: Start Training from Mac
```
1. Open Safari/Chrome → vscode.dev
2. Sign in → Connect to Windows RTX tunnel
3. Open terminal
4. Run: python train_anti_overfitting.py
5. Split terminal (Ctrl+Shift+5)
6. In new terminal: tensorboard --logdir results/experiments/
```

### Example 2: Monitor from iPad
```
1. Open browser → vscode.dev
2. Connect to tunnel
3. Open terminal
4. Run: nvidia-smi -l 2
5. Watch GPU usage live!
```

### Example 3: Emergency Stop Training
```
1. Connect from anywhere
2. Open terminal with running training
3. Press Ctrl+C
4. Model auto-saves checkpoint
```

---

## 🛠️ Troubleshooting

### Can't Find Tunnel
- Make sure VSCode is running on Windows RTX
- Check tunnel is active: `code tunnel status`
- Restart tunnel: `code tunnel --accept-server-license-terms`

### Connection Dropped
- Windows went to sleep → Disable sleep in Power Settings
- VSCode closed → Keep it minimized, not closed
- Internet issues → Check Windows connection

### Port Access for TensorBoard
When running TensorBoard remotely:
1. Start TensorBoard in terminal: `tensorboard --logdir results/experiments/`
2. VSCode will show: "Port 6006 is available"
3. Click "Open in Browser"
4. Or access: `http://localhost:6006` (forwarded automatically!)

---

## 📊 Monitoring Training Remotely

### Real-time GPU Monitoring
```powershell
# Terminal 1: Watch GPU
nvidia-smi -l 1

# Terminal 2: Watch memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Real-time Log Watching
```powershell
# Watch training log
Get-Content results/experiments/*/logs/*_training.log -Wait -Tail 50
```

### TensorBoard Access
```powershell
# Start TensorBoard
tensorboard --logdir results/experiments/ --port 6006

# VSCode automatically forwards port
# Access at: http://localhost:6006
```

---

## ✅ Advantages

- 🌍 **Work from anywhere** - Mac, iPad, phone browser, any device
- 🔒 **Secure** - GitHub authentication, encrypted
- 💻 **Multiple terminals** - Monitor, train, develop simultaneously  
- 🚀 **No VPN needed** - Works through firewalls
- 📱 **Mobile friendly** - Even works on tablets!
- ⚡ **Low latency** - Fast remote editing
- 🔄 **Auto-reconnect** - Survives connection drops

---

## 🎬 Getting Started NOW

### Step 1: On Windows RTX
```
Open VSCode → Ctrl+Shift+P → "Remote Tunnels: Turn on Remote Tunnel Access"
Sign in with GitHub when prompted
Keep VSCode running (can minimize)
```

### Step 2: On Any Other Device
```
Browser: Open https://vscode.dev
Sign in with GitHub
Remote Explorer → Tunnels → Click your Windows machine
Open folder: C:\Users\Naman\emotion-recognition-system
```

### Step 3: Start Training
```
Open terminal (Ctrl+`)
Run: python train_anti_overfitting.py
Done! Training runs on Windows RTX, you control from anywhere!
```

---

That's it! You can now work from your Mac, iPad, or any device, and directly control your Windows RTX training machine! 🎉
