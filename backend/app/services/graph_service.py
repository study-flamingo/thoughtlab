from app.db.neo4j import neo4j_conn
from app.models.nodes import (
    ObservationCreate,
    ObservationUpdate,
    ObservationResponse,
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    HypothesisCreate,
    HypothesisUpdate,
    HypothesisResponse,
    EntityCreate,
    EntityUpdate,
    EntityResponse,
    ConceptCreate,
    ConceptUpdate,
    ConceptResponse,
    RelationshipCreate,
    RelationshipType,
    LinkItem,
)
from app.models.settings import AppSettings, AppSettingsUpdate
from datetime import datetime, UTC
from neo4j.time import DateTime as Neo4jDateTime, Date as Neo4jDate, Time as Neo4jTime
import uuid
from typing import List, Optional, Dict, Any
import json


def _serialize_links(links: Optional[List[LinkItem]]) -> Optional[str]:
    """Serialize links to JSON string for Neo4j storage"""
    if links is None:
        return None
    return json.dumps([link.model_dump() for link in links])


def _deserialize_links(links_json: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Deserialize links from JSON string"""
    if links_json is None:
        return None
    try:
        return json.loads(links_json)
    except (json.JSONDecodeError, TypeError):
        return None


class GraphService:
    """Service for graph database operations"""
    
    async def _ensure_neo4j(self) -> None:
        """Ensure Neo4j driver is connected before queries."""
        if neo4j_conn.driver is None:
            await neo4j_conn.connect()
    
    @staticmethod
    def _json_safe(value: Any, key: str = "") -> Any:
        """Convert values (including Neo4j temporal types) to JSON-safe primitives."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (Neo4jDateTime, Neo4jDate, Neo4jTime)):
            try:
                native = value.to_native()
                # native may be datetime.date/datetime.time/datetime.datetime
                if hasattr(native, "isoformat"):
                    return native.isoformat()
            except Exception:
                # Fallback to string representation
                return str(value)
            return str(value)
        # Deserialize links from JSON string
        if key == "links" and isinstance(value, str):
            return _deserialize_links(value)
        if isinstance(value, dict):
            return {k: GraphService._json_safe(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [GraphService._json_safe(v) for v in value]
        return value
    
    async def create_observation(self, data: ObservationCreate) -> str:
        """Create an observation node, return its ID"""
        await self._ensure_neo4j()
        node_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        query = """
        CREATE (o:Observation {
            id: $id,
            text: $text,
            confidence: $confidence,
            created_at: datetime($created_at),
            concept_names: $concepts,
            links: $links
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
                concepts=data.concept_names or [],
                links=_serialize_links(data.links)
            )
            record = await result.single()
            return record["id"]
    
    async def get_observation(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single observation by ID"""
        await self._ensure_neo4j()
        query = """
        MATCH (o:Observation {id: $id})
        RETURN o
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                node_data = {k: GraphService._json_safe(v, k) for k, v in dict(record["o"]).items()}
                node_data["type"] = "Observation"
                return node_data
            return None
    
    async def get_all_observations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all observation nodes"""
        await self._ensure_neo4j()
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
                node_data = {k: GraphService._json_safe(v, k) for k, v in dict(record["o"]).items()}
                node_data["type"] = "Observation"
                nodes.append(node_data)
            return nodes
    
    async def update_observation(self, node_id: str, data: ObservationUpdate) -> bool:
        """Update an observation node"""
        await self._ensure_neo4j()
        now = datetime.now(UTC)
        updates = []
        params = {"id": node_id}
        
        if data.text is not None:
            updates.append("o.text = $text")
            params["text"] = data.text
        if data.confidence is not None:
            updates.append("o.confidence = $confidence")
            params["confidence"] = data.confidence
        if data.concept_names is not None:
            updates.append("o.concept_names = $concepts")
            params["concepts"] = data.concept_names
        if data.links is not None:
            updates.append("o.links = $links")
            params["links"] = _serialize_links(data.links)
        
        if not updates:
            return False
        
        updates.append("o.updated_at = datetime($updated_at)")
        params["updated_at"] = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        
        query = f"""
        MATCH (o:Observation {{id: $id}})
        SET {', '.join(updates)}
        RETURN o.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None
    
    async def create_source(self, data: SourceCreate) -> str:
        """Create a source node, return its ID"""
        await self._ensure_neo4j()
        node_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        query = """
        CREATE (s:Source {
            id: $id,
            title: $title,
            url: $url,
            source_type: $source_type,
            content: $content,
            published_date: $published_date,
            created_at: datetime($created_at),
            links: $links
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
                },
                links=_serialize_links(data.links)
            )
            record = await result.single()
            return record["id"]
    
    async def update_source(self, node_id: str, data: SourceUpdate) -> bool:
        """Update a source node"""
        await self._ensure_neo4j()
        now = datetime.now(UTC)
        updates = []
        params = {"id": node_id}
        
        if data.title is not None:
            updates.append("s.title = $title")
            params["title"] = data.title
        if data.url is not None:
            updates.append("s.url = $url")
            params["url"] = data.url
        if data.source_type is not None:
            updates.append("s.source_type = $source_type")
            params["source_type"] = data.source_type
        if data.content is not None:
            updates.append("s.content = $content")
            params["content"] = data.content
        if data.published_date is not None:
            updates.append("s.published_date = datetime($published_date)")
            params["published_date"] = {
                "year": data.published_date.year,
                "month": data.published_date.month,
                "day": data.published_date.day,
            }
        if data.links is not None:
            updates.append("s.links = $links")
            params["links"] = _serialize_links(data.links)
        
        if not updates:
            return False
        
        updates.append("s.updated_at = datetime($updated_at)")
        params["updated_at"] = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        
        query = f"""
        MATCH (s:Source {{id: $id}})
        SET {', '.join(updates)}
        RETURN s.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None
    
    async def create_hypothesis(self, data: HypothesisCreate) -> str:
        """Create a hypothesis node, return its ID"""
        await self._ensure_neo4j()
        node_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        query = """
        CREATE (h:Hypothesis {
            id: $id,
            name: $name,
            claim: $claim,
            status: $status,
            created_at: datetime($created_at),
            links: $links
        })
        RETURN h.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=node_id,
                name=data.name,
                claim=data.claim,
                status=data.status,
                created_at={
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "second": now.second,
                },
                links=_serialize_links(data.links)
            )
            record = await result.single()
            return record["id"]
    
    async def update_hypothesis(self, node_id: str, data: HypothesisUpdate) -> bool:
        """Update a hypothesis node"""
        await self._ensure_neo4j()
        now = datetime.now(UTC)
        updates = []
        params = {"id": node_id}
        
        if data.name is not None:
            updates.append("h.name = $name")
            params["name"] = data.name
        if data.claim is not None:
            updates.append("h.claim = $claim")
            params["claim"] = data.claim
        if data.status is not None:
            updates.append("h.status = $status")
            params["status"] = data.status
        if data.links is not None:
            updates.append("h.links = $links")
            params["links"] = _serialize_links(data.links)
        
        if not updates:
            return False
        
        updates.append("h.updated_at = datetime($updated_at)")
        params["updated_at"] = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        
        query = f"""
        MATCH (h:Hypothesis {{id: $id}})
        SET {', '.join(updates)}
        RETURN h.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None
    
    async def create_entity(self, data: EntityCreate) -> str:
        """Create an entity node, return its ID"""
        await self._ensure_neo4j()
        node_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        # Build properties dict
        props = {
            "id": node_id,
            "name": data.name,
            "entity_type": data.entity_type,
            "created_at": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
            },
            "links": _serialize_links(data.links)
        }
        
        # Build query with optional description
        query_parts = [
            "id: $id",
            "name: $name",
            "entity_type: $entity_type",
            "created_at: datetime($created_at)",
            "links: $links"
        ]
        
        if data.description:
            query_parts.append("description: $description")
            props["description"] = data.description
        
        query = f"""
        CREATE (e:Entity {{
            {', '.join(query_parts)}
        }})
        RETURN e.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **props)
            record = await result.single()
            return record["id"]
    
    async def update_entity(self, node_id: str, data: EntityUpdate) -> bool:
        """Update an entity node"""
        await self._ensure_neo4j()
        now = datetime.now(UTC)
        updates = []
        params = {"id": node_id}
        
        if data.name is not None:
            updates.append("e.name = $name")
            params["name"] = data.name
        if data.entity_type is not None:
            updates.append("e.entity_type = $entity_type")
            params["entity_type"] = data.entity_type
        if data.description is not None:
            updates.append("e.description = $description")
            params["description"] = data.description
        if data.properties is not None:
            # Merge with existing properties
            updates.append("e.properties = COALESCE(e.properties, {{}}) + $properties")
            params["properties"] = data.properties
        if data.links is not None:
            updates.append("e.links = $links")
            params["links"] = _serialize_links(data.links)
        
        if not updates:
            return False
        
        updates.append("e.updated_at = datetime($updated_at)")
        params["updated_at"] = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        
        query = f"""
        MATCH (e:Entity {{id: $id}})
        SET {', '.join(updates)}
        RETURN e.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None
    
    async def create_concept(self, data: ConceptCreate) -> str:
        """Create a concept node, return its ID"""
        await self._ensure_neo4j()
        node_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        # Build properties dict
        props = {
            "id": node_id,
            "name": data.name,
            "domain": data.domain,
            "created_at": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
            },
            "links": _serialize_links(data.links)
        }
        
        # Build query with optional description
        query_parts = [
            "id: $id",
            "name: $name",
            "domain: $domain",
            "created_at: datetime($created_at)",
            "links: $links"
        ]
        
        if data.description:
            query_parts.append("description: $description")
            props["description"] = data.description
        
        query = f"""
        CREATE (c:Concept {{
            {', '.join(query_parts)}
        }})
        RETURN c.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **props)
            record = await result.single()
            return record["id"]
    
    async def update_concept(self, node_id: str, data: ConceptUpdate) -> bool:
        """Update a concept node"""
        await self._ensure_neo4j()
        now = datetime.now(UTC)
        updates = []
        params = {"id": node_id}
        
        if data.name is not None:
            updates.append("c.name = $name")
            params["name"] = data.name
        if data.domain is not None:
            updates.append("c.domain = $domain")
            params["domain"] = data.domain
        if data.description is not None:
            updates.append("c.description = $description")
            params["description"] = data.description
        if data.links is not None:
            updates.append("c.links = $links")
            params["links"] = _serialize_links(data.links)
        
        if not updates:
            return False
        
        updates.append("c.updated_at = datetime($updated_at)")
        params["updated_at"] = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        
        query = f"""
        MATCH (c:Concept {{id: $id}})
        SET {', '.join(updates)}
        RETURN c.id as id
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get any node by ID, regardless of type"""
        await self._ensure_neo4j()
        query = """
        MATCH (n {id: $id})
        RETURN labels(n) as labels, n
        LIMIT 1
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                labels = record["labels"]
                node_data = {k: GraphService._json_safe(v, k) for k, v in dict(record["n"]).items()}
                # Set type from first label
                if labels:
                    node_data["type"] = labels[0]
                return node_data
            return None

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships by application id"""
        await self._ensure_neo4j()
        query = """
        MATCH (n {id: $id})
        WITH n LIMIT 1
        DETACH DELETE n
        RETURN 1 as deleted
        """
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            return await result.single() is not None
    
    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two nodes"""
        await self._ensure_neo4j()
        # Ensure each relationship has a stable UUID identifier
        rel_id = str(uuid.uuid4())
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type.value} $props]->(b)
        SET r.id = $rel_id
        RETURN r
        """
        
        props = {k: v for k, v in (properties or {}).items() if v is not None}
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                props=props,
                rel_id=rel_id,
            )
            return await result.single() is not None
    
    async def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by its UUID ID property"""
        await self._ensure_neo4j()
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.id = $rel_id
        RETURN a.id as from_id, b.id as to_id, type(r) as type, 
               r.id as id, properties(r) as props
        LIMIT 1
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, rel_id=relationship_id)
            record = await result.single()
            if record:
                rel_data = {
                    "id": record["id"],
                    "from_id": record["from_id"],
                    "to_id": record["to_id"],
                    "type": record["type"],
                }
                props = dict(record.get("props", {}))
                for key in [
                    "confidence",
                    "notes",
                    "inverse_relationship_type",
                    "inverse_confidence",
                    "inverse_notes",
                    "created_at",
                ]:
                    if key in props:
                        rel_data[key] = GraphService._json_safe(props[key])
                return rel_data
            return None

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by its UUID id property"""
        await self._ensure_neo4j()
        query = """
        MATCH ()-[r]->()
        WHERE r.id = $rel_id
        WITH r LIMIT 1
        DELETE r
        RETURN 1 as deleted
        """
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, rel_id=relationship_id)
            return await result.single() is not None
    
    async def update_relationship(
        self,
        relationship_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update properties of a relationship"""
        await self._ensure_neo4j()
        # Build SET clause dynamically based on provided properties
        set_clauses = []
        params = {"rel_id": relationship_id}
        # Extract potential relationship type change
        new_type = properties.pop("relationship_type", None)
        
        for key, value in properties.items():
            if value is not None:
                param_name = f"prop_{key}"
                set_clauses.append(f"{'nr' if new_type else 'r'}.{key} = ${param_name}")
                params[param_name] = value

        async with neo4j_conn.get_session() as session:
            if new_type:
                # Change relationship type by recreating the relationship and reapplying properties
                set_clause_sql = f"SET {', '.join(set_clauses)}" if set_clauses else ""
                query = f"""
                MATCH (a)-[r]->(b)
                WHERE r.id = $rel_id
                WITH a,b,r,properties(r) AS props
                CREATE (a)-[nr:{new_type}]->(b)
                SET nr = props
                SET nr.id = r.id
                DELETE r
                {set_clause_sql}
                RETURN nr
                """
                result = await session.run(query, **params)
                return await result.single() is not None
            else:
                if not set_clauses:
                    return False
                query = f"""
                MATCH ()-[r]->()
                WHERE r.id = $rel_id
                SET {', '.join(set_clauses)}
                RETURN r
                """
                result = await session.run(query, **params)
                return await result.single() is not None
    
    async def get_node_connections(
        self,
        node_id: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get all nodes connected to a given node within max_depth hops"""
        await self._ensure_neo4j()
        # Note: Cypher does not allow parameterizing the variable-length upper bound.
        # Safely interpolate the integer depth.
        depth = int(max_depth)
        query = f"""
        MATCH path = (n {{id: $id}})-[*1..{depth}]-(connected)
        RETURN DISTINCT connected, relationships(path) as rels
        LIMIT 100
        """
        
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=node_id)
            connections = []
            seen_ids = set()
            
            async for record in result:
                node_data = {k: GraphService._json_safe(v, k) for k, v in dict(record["connected"]).items()}
                
                connected_id = node_data.get("id")
                
                if connected_id and connected_id not in seen_ids:
                    seen_ids.add(connected_id)
                    # Determine node type from labels
                    node_labels = list(record["connected"].labels)
                    node_data["type"] = node_labels[0] if node_labels else "Unknown"
                    
                    connections.append({
                        "node": node_data,
                        "relationships": [dict(r) for r in record["rels"]]
                    })
            
            return connections
    
    async def get_full_graph(self, limit: int = 500, edges_limit: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """Get entire graph structure for visualization"""
        await self._ensure_neo4j()
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
        RETURN a.id as source, b.id as target, type(r) as type, r.id as edge_id, properties(r) as props
        LIMIT $edges_limit
        """
        
        async with neo4j_conn.get_session() as session:
            # Get nodes
            nodes_result = await session.run(nodes_query, limit=limit)
            nodes = []
            async for record in nodes_result:
                node_data = {k: GraphService._json_safe(v, k) for k, v in dict(record["n"]).items()}
                node_labels = list(record["labels"])
                node_data["type"] = node_labels[0] if node_labels else "Unknown"
                nodes.append(node_data)
            
            # Get edges
            edges_result = await session.run(edges_query, edges_limit=edges_limit)
            edges = []
            async for record in edges_result:
                # Extract base edge fields
                edge = {
                    "id": str(record["edge_id"]),
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                }
                # Include selected relationship properties if present
                props = dict(record.get("props", {}))
                for key in [
                    "confidence",
                    "notes",
                    "inverse_relationship_type",
                    "inverse_confidence",
                    "inverse_notes",
                ]:
                    if key in props:
                        edge[key] = GraphService._json_safe(props[key])
                edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}

    async def get_settings(self) -> Dict[str, Any]:
        """Fetch or initialize singleton app settings."""
        await self._ensure_neo4j()
        query = """
        MERGE (s:AppSettings {id: 'app'})
        ON CREATE SET
          s.theme = 'light',
          s.show_edge_labels = true,
          s.default_relation_confidence = 0.8,
          s.layout_name = 'cose',
          s.animate_layout = false
        RETURN s
        """
        async with neo4j_conn.get_session() as session:
            result = await session.run(query)
            record = await result.single()
            node = dict(record["s"])
            # Normalize into AppSettings
            # Decode JSON-encoded complex settings if present
            raw_node_colors = node.get("node_colors", None)
            if isinstance(raw_node_colors, str):
                try:
                    decoded_node_colors = json.loads(raw_node_colors)
                except Exception:
                    decoded_node_colors = None
            else:
                decoded_node_colors = None

            raw_relation_styles = node.get("relation_styles", None)
            if isinstance(raw_relation_styles, str):
                try:
                    decoded_relation_styles = json.loads(raw_relation_styles)
                except Exception:
                    decoded_relation_styles = None
            else:
                decoded_relation_styles = None

            settings = AppSettings(
                id=node.get("id", "app"),
                theme=node.get("theme", "light"),
                show_edge_labels=bool(node.get("show_edge_labels", True)),
                default_relation_confidence=float(node.get("default_relation_confidence", 0.8)),
                layout_name=node.get("layout_name", "cose"),
                animate_layout=bool(node.get("animate_layout", False)),
                node_colors=decoded_node_colors or AppSettings().node_colors,
                relation_styles=decoded_relation_styles or AppSettings().relation_styles,
            )
            return settings.model_dump()

    async def update_settings(self, update: AppSettingsUpdate) -> Dict[str, Any]:
        """Update settings fields atomically, returning updated settings."""
        await self._ensure_neo4j()
        updates: Dict[str, Any] = {}
        if update.theme is not None:
            updates["theme"] = update.theme
        if update.show_edge_labels is not None:
            updates["show_edge_labels"] = update.show_edge_labels
        if update.default_relation_confidence is not None:
            updates["default_relation_confidence"] = update.default_relation_confidence
        if update.layout_name is not None:
            updates["layout_name"] = update.layout_name
        if update.animate_layout is not None:
            updates["animate_layout"] = update.animate_layout
        if update.node_colors is not None:
            # Store as JSON string to comply with Neo4j property constraints
            try:
                updates["node_colors"] = json.dumps(update.node_colors)
            except Exception:
                # Fallback to empty dict JSON
                updates["node_colors"] = json.dumps({})
        if update.relation_styles is not None:
            # Convert RelationStyle models to dicts, then JSON-encode
            try:
                relation_styles_dict = {
                    k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in update.relation_styles.items()
                }
                updates["relation_styles"] = json.dumps(relation_styles_dict)
            except Exception:
                updates["relation_styles"] = json.dumps({})

        if not updates:
            # No-op; just return current settings
            return await self.get_settings()

        set_clauses = []
        params: Dict[str, Any] = {}
        for key, value in updates.items():
            param_name = f"p_{key}"
            set_clauses.append(f"s.{key} = ${param_name}")
            params[param_name] = value

        query = f"""
        MERGE (s:AppSettings {{id: 'app'}})
        SET {', '.join(set_clauses)}
        RETURN s
        """
        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            node = dict(record["s"])
            # Decode JSON-encoded complex settings if present
            raw_node_colors = node.get("node_colors", None)
            if isinstance(raw_node_colors, str):
                try:
                    decoded_node_colors = json.loads(raw_node_colors)
                except Exception:
                    decoded_node_colors = None
            else:
                decoded_node_colors = None

            raw_relation_styles = node.get("relation_styles", None)
            if isinstance(raw_relation_styles, str):
                try:
                    decoded_relation_styles = json.loads(raw_relation_styles)
                except Exception:
                    decoded_relation_styles = None
            else:
                decoded_relation_styles = None

            settings = AppSettings(
                id=node.get("id", "app"),
                theme=node.get("theme", "light"),
                show_edge_labels=bool(node.get("show_edge_labels", True)),
                default_relation_confidence=float(node.get("default_relation_confidence", 0.8)),
                layout_name=node.get("layout_name", "cose"),
                animate_layout=bool(node.get("animate_layout", False)),
                node_colors=decoded_node_colors or AppSettings().node_colors,
                relation_styles=decoded_relation_styles or AppSettings().relation_styles,
            )
            return settings.model_dump()


# Global service instance
graph_service = GraphService()
