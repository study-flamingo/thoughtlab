#!/bin/sh
# Docker entrypoint for Railway deployment
# Substitutes PORT environment variable into nginx config

set -e

# Default to port 80 if not set (local dev)
export PORT=${PORT:-80}

# Substitute environment variables in nginx config
envsubst '${PORT}' < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf.tmp
mv /etc/nginx/conf.d/default.conf.tmp /etc/nginx/conf.d/default.conf

# Execute the main command
exec "$@"
