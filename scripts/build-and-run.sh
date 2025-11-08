#!/usr/bin/env bash

# Fully builds everything from scratch, you probably want to do this per container then runs daemoized if successful.
#
# You might find it useful to run this without -d, as it will print logs live. Then use another terminal to exec into containers
sudo docker compose build --no-cache && \
  sudo docker compose up -d
