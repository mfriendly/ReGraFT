#!/bin/bash

# Define the paths to the scripts
SCRIPT_DIR="E:/py/reddump/23-09-03_MY/MYEDA/ReGraFT_JBHI제출용"
SCRIPT1="$SCRIPT_DIR/git_upload_dir1.sh"
SCRIPT2="$SCRIPT_DIR/git_upload_dir2.sh"

# Ensure both scripts are executable
chmod +x "$SCRIPT1"
chmod +x "$SCRIPT2"

# Run both scripts in background
"$SCRIPT1" &
"$SCRIPT2" &

# Wait for both scripts to finish
wait

echo "Both scripts have completed execution."
