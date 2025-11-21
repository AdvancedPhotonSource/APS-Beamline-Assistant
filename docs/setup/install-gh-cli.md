# Installing GitHub CLI (gh)

## On macOS (Homebrew)

```bash
brew install gh
```

## After Installation

```bash
# Authenticate with GitHub
gh auth login

# Create PR from command line
cd /Users/b324240/Git/beamline-assistant-dev
gh pr create --title "Modular Architecture v2.0 - Complete Platform Redesign" \
  --body "Complete redesign with modular architecture, 20+ MIDAS tools, and comprehensive documentation." \
  --base main \
  --head pawan-modular-v2
```

This makes creating PRs much easier!
