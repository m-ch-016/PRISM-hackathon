#!/bin/bash

# Run the server on different ports in the background
python3 server.py -p 8001 &  # Run server on port 8001
python3 server.py -p 8002 &  # Run server on port 8002
python3 server.py -p 8003 &  # Run server on port 8003
python3 server.py -p 8004 &  # Run server on port 8004
python3 server.py -p 8005 &  # Run server on port 8005
python3 server.py -p 8006 &  # Run server on port 8006

# Wait for all background processes to finish
wait
