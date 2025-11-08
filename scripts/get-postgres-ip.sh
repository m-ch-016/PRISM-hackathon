#!/usr/bin/env bash
sudo docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(sudo docker ps | grep -i postgres | grep -Eo '^([a-f0-9]+)')
