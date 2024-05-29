
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
# git reset HEAD "$REPO_DIR/data/x_data_pkl/*.pkl"

# Force push to main branch on origin
git push origin main --force

echo "Changes pushed to remote."
