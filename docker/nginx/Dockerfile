FROM nginx:1.25.3

# Install required dependencies
RUN apt-get update \
    && apt-get -y install lsb-release libssl-dev ca-certificates curl openssl cron

# Nginx folders
RUN mkdir -p /var/cache/nginx/client_temp \
	&& mkdir -p /var/cache/nginx/proxy_temp \
	&& mkdir -p /var/cache/nginx/fastcgi_temp \
	&& mkdir -p /var/cache/nginx/uwsgi_temp \
	&& mkdir -p /var/cache/nginx/scgi_temp

# Expose ports
EXPOSE 80 443
ADD entry.sh /usr/local/bin/entrypoint.sh
RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]

RUN mkdir -p /var/www
COPY site-conf/ /home/comfyuser/docker/nginx/site-conf
COPY nginx.conf /home/comfyuser/docker/nginx/nginx.conf
COPY ssl/self-signed/cert.pem /home/comfyuser/docker/nginx/ssl/cert.pem
COPY ssl/self-signed/privkey.pem /home/comfyuser/docker/nginx/ssl/privkey.pem

WORKDIR /home/comfyuser
# Start Nginx
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]