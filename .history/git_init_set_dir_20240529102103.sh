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
git commit -m "Track .pkl files with Git LFS"

echo "Initialization complete."
Script 2: Changes and Pushing to Remote
This script handles the adding of files, committing changes, and pushing them to the remote repository.

bash
코드 복사
#!/bin/bash

# Set the directory variable
REPO_DIR="E:/PY/reddump/23-09-03_MY/MYEDA/ReGraft_JBHI제출용"

# Navigate to the directory
cd "$REPO_DIR"

# Add all files and commit changes
git add .
git commit -m "Committing all changes"

# Remove __pycache__ directories recursively
find . -name "__pycache__" -type d -exec rm -rf {} +

# Remove cached files, reset and re-add all files
git rm -r --cached .
git reset
git add -A
git commit -m "Reset cache and re-add all files"

# Handle specific data exclusion
git reset HEAD "$REPO_DIR/data/x_data_pkl/*.pkl"

# Force push to main branch on origin
git push origin main --force

echo "Changes pushed to remote."
Usage Notes:

Directory Path: Ensure the REPO_DIR variable correctly reflects the path to your repository. Adjust as needed for different environments.
Permissions: These scripts should be executable. You can make them executable by running chmod +x scriptname.sh.