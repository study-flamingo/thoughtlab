#!/bin/bash
# Restore Neo4j named volumes from a tar.gz created by backup_neo4j.sh
#
# Usage:
#   ./scripts/restore_neo4j.sh path/to/backup.tar.gz
# Notes:
# - Stop the neo4j container before restoring: docker-compose stop neo4j
# - This will overwrite existing data/logs in the volumes

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <backup-file.tar.gz>"
  exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "❌ Backup file not found: $BACKUP_FILE"
  exit 1
fi

echo "🧰 Restoring Neo4j from: $BACKUP_FILE"
echo "ℹ️  Ensure 'neo4j' service is stopped: docker-compose stop neo4j"

docker run --rm \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -v "$(pwd)":/backup \
  alpine:3 \
  sh -c "cd / && tar -xzf /backup/$(basename "$BACKUP_FILE")"

echo "✅ Restore complete. Start neo4j when ready: docker-compose up -d neo4j"

