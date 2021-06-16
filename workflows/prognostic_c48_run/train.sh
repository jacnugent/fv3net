#!/bin/bash

set -e

echo "Pulling data with dvc" > /dev/stderr
dvc pull data/training data/validation
echo "Starting Training" > /dev/stderr
guild run train $@
echo "Syncing to $REMOTE_GUILD_HOME" > /dev/stderr
gsutil -m rsync -r "$GUILD_HOME/runs" "$REMOTE_GUILD_HOME/runs"
