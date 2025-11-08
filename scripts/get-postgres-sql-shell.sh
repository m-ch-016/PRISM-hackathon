#!/usr/bin/env bash

PIP="$(./get-postgres-ip.sh)"
echo "Postgres IP: $PIP"
echo "Use the password from the docker compose yaml file"
echo "Run \dt for tables and write <SQL>; with the semicolon to run sql (google otherwise)"
psql -h $PIP -U postgres -d prism
