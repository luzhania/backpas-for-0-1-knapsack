#!/bin/bash

SESSION_NAME="backbone"
# Command to load conda, activate the environment, and run the script
COMMAND='eval "$(conda shell.bash hook)" && conda activate backpas && ./backpas/src/0_run_backbone_extraction.sh ./guroback/guroback ./backpas/dataset/14_bounded_strongly_correlated "*.opb"'

echo "Starting watcher for session: $SESSION_NAME..."

while true; do
    # Verify if the tmux session does NOT exist
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "[$(date)] Session $SESSION_NAME does not exist or was closed (possible Out of Memory). Restarting..."
        # Start detached session (-d)
        tmux new-session -d -s "$SESSION_NAME" "bash -c '$COMMAND'"
    fi
    # Wait 300 seconds before checking again
    sleep 300
done