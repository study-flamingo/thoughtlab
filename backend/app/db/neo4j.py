from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.config import settings
from typing import Optional


class Neo4jConnection:
    """Neo4j database connection manager"""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
    
    async def connect(self):
        """Connect to Neo4j database"""
        if self.driver is None:
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Verify connection
            await self.driver.verify_connectivity()
            print("Connected to Neo4j")
    
    async def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            print("Disconnected from Neo4j")
    
    def get_session(self):
        """Get a Neo4j session"""
        if self.driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self.driver.session()


# Global connection instance
neo4j_conn = Neo4jConnection()
