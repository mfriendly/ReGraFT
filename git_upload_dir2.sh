#!/bin/bash


REPO_DIR="E:/PY/reddump/23-09-03_MY/MYEDA/ReGraft_JBHI제출용"


cd "$REPO_DIR"


echo "*.sh" >> .gitignore
echo "results/" >> .gitignore
echo ".history/" >> .gitignore
git add .gitignore
git commit -m "Updated .gitignore to exclude .sh files and temp folder"


git add .
git commit -m "Committing all changes"


find . -name "__pycache__" -type d -exec rm -rf {} +


git rm -r --cached .


git reset


git add -A
git commit -m "Reset cache and re-add all files"





git push origin main --force

echo "Changes pushed to remote."
