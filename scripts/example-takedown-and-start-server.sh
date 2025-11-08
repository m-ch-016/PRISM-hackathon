# serves as an example on reloading changes to a specific server.
sudo docker compose down prism-server && sudo docker build prism-server --no-cache && sudo docker compose up 
