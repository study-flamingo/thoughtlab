from app.db.neo4j import neo4j_conn
from app.models.nodes import (
    ObservationCreate,
    ObservationResponse,
    SourceCreate,
    SourceResponse,
    HypothesisCreate,
    HypothesisResponse,
    RelationshipCreate,
    RelationshipType,
)
from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any


class GraphService:
    """Service for graph database operations"""
    
    async def create_observation(self, data: ObservationCreate) -> str:
        """Create an observation node, return its ID"""
        node_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        CREATE (o:Observation {
            id: $id,
            text: $text,
            confidence: $confidence,
            created_at: datetime($created_at),
            concept_names: $concepts
        })
        RETURN o.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=node_id,
                text=data.text,
                confidence=data.confidence,
                created_at={
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "second": now.second,
                },
                concepts=data.concept_names or []
            )
            record = await result.single()
            return record["id"]
    
    async def get_observation(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single observation by ID"""
        query = """
        MATCH (o:Observation {id: $id})
        RETURN o
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                node_data = dict(record["o"])
                # Convert datetime objects to ISO strings for JSON serialization
                for key, value in node_data.items():
                    if isinstance(value, datetime):
                        node_data[key] = value.isoformat()
                node_data["type"] = "Observation"
                return node_data
            return None
    
    async def get_all_observations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all observation nodes"""
        query = """
        MATCH (o:Observation)
        RETURN o
        ORDER BY o.created_at DESC
        LIMIT $limit
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, limit=limit)
            nodes = []
            async for record in result:
                node_data = dict(record["o"])
                # Convert datetime objects to ISO strings for JSON serialization
                for key, value in node_data.items():
                    if isinstance(value, datetime):
                        node_data[key] = value.isoformat()
                node_data["type"] = "Observation"
                nodes.append(node_data)
            return nodes
    
    async def create_source(self, data: SourceCreate) -> str:
        """Create a source node, return its ID"""
        node_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        CREATE (s:Source {
            id: $id,
            title: $title,
            url: $url,
            source_type: $source_type,
            content: $content,
            published_date: $published_date,
            created_at: datetime($created_at)
        })
        RETURN s.id as id
        """
        
        published_date = None
        if data.published_date:
            published_date = {
                "year": data.published_date.year,
                "month": data.published_date.month,
                "day": data.published_date.day,
            }
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=node_id,
                title=data.title,
                url=data.url,
                source_type=data.source_type,
                content=data.content,
                published_date=published_date,
                created_at={
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "second": now.second,
                }
            )
            record = await result.single()
            return record["id"]
    
    async def create_hypothesis(self, data: HypothesisCreate) -> str:
        """Create a hypothesis node, return its ID"""
        node_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        CREATE (h:Hypothesis {
            id: $id,
            claim: $claim,
            status: $status,
            created_at: datetime($created_at)
        })
        RETURN h.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=node_id,
                claim=data.claim,
                status=data.status,
                created_at={
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "second": now.second,
                }
            )
            record = await result.single()
            return record["id"]
    
    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes"""
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type.value} $props]->(b)
        RETURN r
        """
        
        props = properties or {}
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                props=props
            )
            return await result.single() is not None
    
    async def get_node_connections(
        self,
        node_id: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get all nodes connected to a given node within max_depth hops"""
        query = """
        MATCH path = (n {id: $id})-[*1..$depth]-(connected)
        RETURN DISTINCT connected, relationships(path) as rels
        LIMIT 100
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id, depth=max_depth)
            connections = []
            seen_ids = set()
            
            async for record in result:
                node_data = dict(record["connected"])
                # Convert datetime objects to ISO strings for JSON serialization
                for key, value in node_data.items():
                    if isinstance(value, datetime):
                        node_data[key] = value.isoformat()
                
                node_id = node_data.get("id")
                
                if node_id and node_id not in seen_ids:
                    seen_ids.add(node_id)
                    # Determine node type from labels
                    node_labels = list(record["connected"].labels)
                    node_data["type"] = node_labels[0] if node_labels else "Unknown"
                    
                    connections.append({
                        "node": node_data,
                        "relationships": [dict(r) for r in record["rels"]]
                    })
            
            return connections
    
    async def get_full_graph(self, limit: int = 500) -> Dict[str, List[Dict[str, Any]]]:
        """Get entire graph structure for visualization"""
        nodes_query = """
        MATCH (n)
        WHERE n:Observation OR n:Hypothesis OR n:Source OR n:Entity OR n:Concept
        RETURN n, labels(n) as labels
        LIMIT $limit
        """
        
        edges_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:Observation OR a:Hypothesis OR a:Source OR a:Entity OR a:Concept)
          AND (b:Observation OR b:Hypothesis OR b:Source OR b:Entity OR b:Concept)
        RETURN a.id as source, b.id as target, type(r) as type, id(r) as edge_id
        LIMIT 1000
        """
        
        async with neo4j_conn.get_session() as session:
            # Get nodes
            nodes_result = await session.run(nodes_query, limit=limit)
            nodes = []
            async for record in nodes_result:
                node_data = dict(record["n"])
                # Convert datetime objects to ISO strings for JSON serialization
                for key, value in node_data.items():
                    if isinstance(value, datetime):
                        node_data[key] = value.isoformat()
                node_labels = list(record["labels"])
                node_data["type"] = node_labels[0] if node_labels else "Unknown"
                nodes.append(node_data)
            
            # Get edges
            edges_result = await session.run(edges_query)
            edges = []
            async for record in edges_result:
                edges.append({
                    "id": str(record["edge_id"]),
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"]
                })
        
        return {"nodes": nodes, "edges": edges}


# Global service instance
graph_service = GraphService()
