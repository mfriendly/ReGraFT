bash
코드 복사
#!/bin/bash

# Set the directory variable
REPO_DIR="E:/PY/reddump/23-09-03_MY/MYEDA/ReGraft_JBHI제출용"

# Initialize Git repository
git init

# Configure safe directory
git config --global --add safe.directory "$REPO_DIR"

# Set global user configurations
git config --global user.email "minkkim1228@gmail.com"
git config --global user.name "mfriendly"
git config --global credential.helper store

# Initialize bare repository and set remotes
cd "$REPO_DIR"
git init --bare
git remote add origin https://github.com/mfriendly/ReGraFT
git remote add Y "$REPO_DIR"

# Track large files with Git LFS
git lfs install
git lfs track "*.pkl"
git add .gitattributes
# git commit -m "Track .pkl files with Git LFS"

echo "Initialization complete."
