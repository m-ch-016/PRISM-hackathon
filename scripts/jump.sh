#!/usr/bin/env bash

# This script spawns a shell into a container which makes it easy to
# run things with context.
#
# Use as ./jump.sh <container name>
#
# You can find container names via running the following from project root
# sudo docker compose ps

if [[ $# -ne 1 ]]; then 
	echo -e "Usage: $0 <search string>"
	exit 1
fi
CONTAINER="$(sudo docker compose ps | grep -v name | cut -d' ' -f1 | grep $1 | head -n1)"
echo "Jumping to $CONTAINER"
sudo docker exec -it "$CONTAINER" bash
