#!/bin/bash
mkdir -p docker/nginx/ssl/self-signed
openssl req \
    -new \
    -newkey rsa:4096 \
    -days 365 \
    -nodes \
    -x509 \
    -subj "/C=XX/ST=XXX/L=XXX/O=XXX/CN=localhost" \
    -keyout docker/nginx/ssl/self-signed/privkey.pem \
    -out docker/nginx/ssl/self-signed/cert.pem