#!/bin/sh

# Copy the configuration files to their respective locations
cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf
echo "Success: Configuration enabled for ${SITE_DOMAIN}."

# Execute the CMD provided in the Dockerfile
exec "$@"