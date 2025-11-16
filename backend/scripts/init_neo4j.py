#!/usr/bin/env python3
"""
Initialize Neo4j database with indexes and constraints.
Run this script after Neo4j is up and running.
"""
import asyncio
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()


async def init_neo4j():
    """Initialize Neo4j with indexes and constraints"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "research_graph_password")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    init_queries = [
        # Unique constraints on IDs
        "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        
        # Text indexes
        "CREATE TEXT INDEX observation_text IF NOT EXISTS FOR (o:Observation) ON o.text",
        "CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS FOR (h:Hypothesis) ON h.claim",
        "CREATE TEXT INDEX source_title IF NOT EXISTS FOR (s:Source) ON s.title",
        
        # Temporal indexes
        "CREATE INDEX observation_created IF NOT EXISTS FOR (o:Observation) ON o.created_at",
        "CREATE INDEX hypothesis_created IF NOT EXISTS FOR (h:Hypothesis) ON h.created_at",
        "CREATE INDEX source_created IF NOT EXISTS FOR (s:Source) ON s.created_at",
    ]
    
    try:
        async with driver.session() as session:
            for query in init_queries:
                try:
                    await session.run(query)
                    print(f"✓ {query[:50]}...")
                except Exception as e:
                    print(f"✗ Error executing: {query[:50]}...")
                    print(f"  {str(e)}")
            
            # Verify
            result = await session.run("SHOW INDEXES")
            indexes = [record async for record in result]
            print(f"\n✓ Created {len(indexes)} indexes")
            
            result = await session.run("SHOW CONSTRAINTS")
            constraints = [record async for record in result]
            print(f"✓ Created {len(constraints)} constraints")
            
    finally:
        await driver.close()


if __name__ == "__main__":
    print("Initializing Neo4j database...")
    asyncio.run(init_neo4j())
    print("Done!")
