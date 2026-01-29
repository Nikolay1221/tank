#!/bin/bash
# start_training.sh

# Navigate to project root (assumes script is in /linux/ subdir or project root)
# Let's assume the user puts everything in ~/battle_city_ai_new
cd "$(dirname "$0")/.." || cd "$(dirname "$0")"

# Activate environment if it exists (optional, adjust valid path)
# source venv/bin/activate

# Set PYTHONPATH to current directory so python -m works
export PYTHONPATH=$PYTHONPATH:.

# Clean up any potential stale locks (optional)
# rm -f /tmp/battle_city.lock

# Run training
# unbuffered output (-u) to see logs in real time
exec python3 -u -m python.train
