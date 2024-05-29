#!/bin/bash


REPO_DIR="E:/PY/reddump/23-09-03_MY/MYEDA/ReGraft_JBHI제출용"


git init


git config --global --add safe.directory "$REPO_DIR"


git config --global user.email "minkkim1228@gmail.com"
git config --global user.name "mfriendly"
git config --global credential.helper store


cd "$REPO_DIR"
git init --bare
git remote add origin https://github.com/mfriendly/ReGraFT
git remote add Y "$REPO_DIR"


git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track .pkl files with Git LFS"

echo "Initialization complete."
