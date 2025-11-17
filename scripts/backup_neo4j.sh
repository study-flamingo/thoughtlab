#!/bin/bash
# Backup Neo4j named volumes to a tar.gz in the current directory.
#
# Usage:
#   ./scripts/backup_neo4j.sh [output-file.tar.gz]
# Example:
#   ./scripts/backup_neo4j.sh neo4j-backup-$(date +%Y%m%d-%H%M%S).tar.gz
#
# Notes:
# - Requires docker and the named volumes defined in docker-compose.yml
# - Does NOT require the neo4j container to be running

set -e

OUTPUT_NAME="${1:-neo4j-backup-$(date +%Y%m%d-%H%M%S).tar.gz}"

echo "📦 Creating Neo4j backup: $OUTPUT_NAME"
docker run --rm \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -v "$(pwd)":/backup \
  alpine:3 \
  sh -c "cd / && tar -czf /backup/$OUTPUT_NAME data logs"

echo "✅ Backup created: $OUTPUT_NAME"

