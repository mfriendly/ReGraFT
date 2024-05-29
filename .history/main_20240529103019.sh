#!/bin/bash


SCRIPT_DIR="E:/py/reddump/23-09-03_MY/MYEDA/ReGraFT_JBHI제출용"
SCRIPT1="$SCRIPT_DIR/git_init_set_dir.sh"
SCRIPT2="$SCRIPT_DIR/git_upload_dir2.sh"


chmod +x "$SCRIPT1"
chmod +x "$SCRIPT2"


"$SCRIPT1" &
"$SCRIPT2" &


wait

echo "Both scripts have completed execution."
