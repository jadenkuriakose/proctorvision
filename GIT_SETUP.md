# Git Repository Setup Instructions

## Step 1: Initialize Git Repository

```bash
cd /Users/jadenkuriakose/proctorvision
git init
git add .
git commit -m "Initial commit: ProctorVision proctoring system"
```

## Step 2: Create Remote Repository

### Option A: GitHub
1. Go to https://github.com/new
2. Create a new repository named `proctorvision`
3. **DO NOT** initialize with README, gitignore, or license (we have them)
4. Copy the repository URL (HTTPS or SSH)

### Option B: GitLab / Gitea / Custom Server
Follow similar steps for your preferred platform.

## Step 3: Add Remote and Push

```bash
# Add remote (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/proctorvision.git

# Verify remote was added
git remote -v

# Push to remote
git branch -M main
git push -u origin main
```

## Step 4: Verify Success

```bash
# Check that files are on remote
git log --oneline

# View remote status
git status
```

## Alternative: One-Command Setup

If you prefer a single command flow:

```bash
cd /Users/jadenkuriakose/proctorvision && \
git init && \
git add . && \
git commit -m "Initial commit: ProctorVision proctoring system" && \
git branch -M main && \
git remote add origin https://github.com/YOUR_USERNAME/proctorvision.git && \
git push -u origin main
```

## What Gets Ignored

The `.gitignore` file ensures these are NOT committed:
- Virtual environments (`venv/`, `.venv/`)
- Model files (`*.pt`, `*.onnx`) - download on demand
- Generated frames (`frames/`, `*.ppm`)
- Build artifacts (`camera` binary, `*.o`)
- IDE files (`.vscode/`, `.idea/`)
- Logs and temporary files
- OS files (`.DS_Store`)

This keeps the repo small (~500KB instead of 1GB+) and deployable.

## Next Steps After Push

Once pushed, you can:

1. **Create development branch:**
   ```bash
   git checkout -b feature/improvements
   ```

2. **Track changes:**
   ```bash
   git status
   git diff
   git add <file>
   git commit -m "Description"
   git push -u origin feature/improvements
   ```

3. **Pull latest changes:**
   ```bash
   git pull origin main
   ```
