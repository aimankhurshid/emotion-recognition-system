#!/bin/bash

echo "================================================================================"
echo "EMOTION RECOGNITION SYSTEM - MAC SETUP"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're already in the repo
if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Already in repository directory${NC}"
else
    echo -e "${YELLOW}Cloning repository...${NC}"
    git clone https://github.com/aimankhurshid/emotion-recognition-system.git
    cd emotion-recognition-system || exit
fi

echo ""
echo "================================================================================"
echo "STEP 1: Git Configuration"
echo "================================================================================"

# Configure Git
read -p "Enter your GitHub username [aimankhurshid]: " git_username
git_username=${git_username:-aimankhurshid}

read -p "Enter your GitHub email: " git_email

git config user.name "$git_username"
git config user.email "$git_email"

echo -e "${GREEN}✓ Git configured${NC}"

echo ""
echo "================================================================================"
echo "STEP 2: Python Environment"
echo "================================================================================"

# Check Python version
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $python_version${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) [y]: " create_venv
create_venv=${create_venv:-y}

if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
fi

# Install dependencies
read -p "Install Python dependencies? (y/n) [y]: " install_deps
install_deps=${install_deps:-y}

if [ "$install_deps" = "y" ]; then
    echo "Installing dependencies..."
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

echo ""
echo "================================================================================"
echo "STEP 3: VSCode Configuration"
echo "================================================================================"

# Check if VSCode is installed
if command -v code &> /dev/null; then
    echo -e "${GREEN}✓ VSCode found${NC}"
    
    read -p "Install recommended VSCode extensions? (y/n) [y]: " install_extensions
    install_extensions=${install_extensions:-y}
    
    if [ "$install_extensions" = "y" ]; then
        echo "Installing VSCode extensions..."
        code --install-extension ms-python.python
        code --install-extension ms-python.vscode-pylance
        code --install-extension ms-toolsai.jupyter
        code --install-extension eamodio.gitlens
        code --install-extension ms-vscode-remote.remote-ssh
        echo -e "${GREEN}✓ Extensions installed${NC}"
    fi
    
    read -p "Open project in VSCode? (y/n) [y]: " open_vscode
    open_vscode=${open_vscode:-y}
    
    if [ "$open_vscode" = "y" ]; then
        code .
        echo -e "${GREEN}✓ Opened in VSCode${NC}"
    fi
else
    echo -e "${YELLOW}⚠ VSCode not found. Install from: https://code.visualstudio.com${NC}"
fi

echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "📁 Project Location: $(pwd)"
echo ""
echo "🎯 Next Steps:"
echo "   1. Review documentation: cat MAC_SETUP_GUIDE.md"
echo "   2. Check training strategy: cat TRAINING_STRATEGY.md"
echo "   3. View current results: python visualize_training.py"
echo ""
echo "💻 Useful Commands:"
echo "   • Pull latest from Windows: git pull origin main"
echo "   • View training results: python visualize_experiment.py --compare"
echo "   • Push your changes: git commit -m 'Your message [Mac-Research]' && git push"
echo ""
echo "📚 Documentation:"
echo "   • MAC_SETUP_GUIDE.md - Complete Mac setup guide"
echo "   • GIT_WORKFLOW.md - Git workflow for Windows/Mac"
echo "   • TRAINING_STRATEGY.md - Training documentation"
echo ""
echo "================================================================================"
echo ""

# Create a quick reference card
cat > QUICK_START_MAC.md << 'EOF'
# Quick Start - Mac

## Daily Workflow

### Before Starting Work
```bash
git pull origin main
```

### After Making Changes
```bash
git add .
git status
git commit -m "Your message [Mac-Research]"
git push origin main
```

## Useful Commands

### View Training Results
```bash
python visualize_training.py
python visualize_experiment.py --compare
```

### Analyze Data
```bash
jupyter notebook notebooks/
```

### Run Tests
```bash
python -m pytest tests/
```

### Update Documentation
```bash
# Edit markdown files
code README.md
```

## Sync with Windows RTX

### Pull Latest Training Results
```bash
git pull origin main
# View new results
ls -la results/experiments/
```

### Access Windows via SSH (if configured)
```bash
ssh windows-rtx
# or in VSCode: Cmd+Shift+P → "Remote-SSH: Connect to Host"
```

## Commit Message Convention

Format: `<type>: <description> [Mac-Research]`

Examples:
- `docs: Update research findings [Mac-Research]`
- `analysis: Add performance comparison plots [Mac-Research]`
- `refactor: Improve code structure [Mac-Research]`

## Help

- Full guide: `cat MAC_SETUP_GUIDE.md`
- Git workflow: `cat GIT_WORKFLOW.md`
- Training docs: `cat TRAINING_STRATEGY.md`
EOF

echo -e "${GREEN}✓ Created QUICK_START_MAC.md for quick reference${NC}"
echo ""
