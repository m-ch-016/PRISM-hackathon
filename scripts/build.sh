#!/usr/bin/env bash

# Two cases:
# 1. NO ARGUMENTS PASSED
#    --> Fully builds everything from scratch, you probably want to do this per container.
# 2. CONTAINER NAME(S) PASSED
#    --> Fully builds specific container(s) from scratch.

if [[ $# -eq 0 ]]; then 
  sudo docker compose build --no-cache
  echo "0"
elif [[ $# -eq 1 ]]; then
  echo "building $1"
  sudo docker compose build "$1" --no-cache
else
  echo "Usage $0 (search string)"
fi
