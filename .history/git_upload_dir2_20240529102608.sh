#!/bin/bash

# Define the repository directory
REPO_DIR="E:/PY/reddump/23-09-03_MY/MYEDA/ReGraft_JBHI제출용"

# Navigate to the directory
cd "$REPO_DIR"

# Ensure .gitignore is set to ignore .sh files
echo "*.sh" >> .gitignore
git add .gitignore
git commit -m "Update .gitignore to exclude .sh files"

# Add all files except those excluded by .gitignore
git add .
git commit -m "Committing all changes"

# Remove __pycache__ directories recursively
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clear cached entries, this removes all files from the staging area
git rm -r --cached .

# Reset the index to unstage all files, might be redundant but ensures no leftovers
git reset

# Re-add all files as per current .gitignore rules
git add -A
git commit -m "Reset cache and re-add all files"

# Handle specific data exclusion
# git reset HEAD "$REPO_DIR/data/x_data_pkl/*.pkl"

# Force push to the main branch on the origin
git push origin main --force

echo "Changes pushed to remote."
