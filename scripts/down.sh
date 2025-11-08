#!/usr/bin/env bash

# This kills a specific docker container that matches the search string passed.

if [[ $# -ne 1 ]]; then 
	echo -e "Usage: $0 <search string>"
	exit 1
fi
CONTAINER="$(sudo docker compose ps | grep -v name | cut -d' ' -f1 | grep $1 | head -n1)"
echo "Killing $CONTAINER"
sudo docker compose down "$CONTAINER"

