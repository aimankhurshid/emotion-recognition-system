# Git Configuration Guide

## Setting Up Machine-Specific Commit Signatures

### On Windows RTX (Training Machine)
```powershell
# Configure Git to add [Windows-RTX] suffix
git config --local commit.template .gitmessage-windows

# Or add manually to each commit:
git commit -m "Your commit message [Windows-RTX]"
```

### On Mac (Research Machine)
```bash
# Configure Git to add [Mac-Research] suffix
git config --local commit.template .gitmessage-mac

# Or add manually to each commit:
git commit -m "Your commit message [Mac-Research]"
```

## Quick Push Scripts

### Windows: `push_to_github.ps1`
```powershell
.\push_to_github.ps1
```

### Mac: Create `push_to_github.sh`
```bash
#!/bin/bash
git add .
git commit -m "Your message [Mac-Research]"
git push origin main
```

## Commit Message Convention

**Format**: `<action>: <description> [<machine>]`

**Examples:**
- `feat: Add new training scheduler [Windows-RTX]`
- `fix: Update data loader bug [Mac-Research]`
- `train: Complete epoch 50 training run [Windows-RTX]`
- `docs: Update research findings [Mac-Research]`
- `refactor: Optimize model architecture [Mac-Research]`
- `experiment: Anti-overfitting training (dropout 0.6) [Windows-RTX]`

## Types of Changes by Machine

### Windows RTX (Training)
- `train:` - Training runs
- `experiment:` - Experimental training configurations
- `perf:` - Performance optimizations
- `model:` - Model architecture changes
- `results:` - Training results and logs

### Mac (Research)
- `docs:` - Documentation and research notes
- `research:` - Research experiments and analysis
- `analysis:` - Data analysis and visualizations
- `refactor:` - Code refactoring
- `fix:` - Bug fixes

## Auto-tagging Setup

Add this to your PowerShell profile (Windows) or `.bashrc`/`.zshrc` (Mac):

### Windows PowerShell Profile
```powershell
# Add to: $PROFILE
function gitc {
    param([string]$message)
    git commit -m "$message [Windows-RTX]"
}

function gitp {
    git push origin main
}
```

Usage: `gitc "Add training script"; gitp`

### Mac Bash/Zsh
```bash
# Add to ~/.bashrc or ~/.zshrc
alias gitc='git commit -m'
gitcm() {
    git commit -m "$1 [Mac-Research]"
}
alias gitp='git push origin main'
```

Usage: `gitcm "Update research notes"; gitp`
