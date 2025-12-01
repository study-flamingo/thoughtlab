"""Activity Feed service for managing system events and user interactions.

The Activity Feed is the central place where users see:
- What's happening in their knowledge graph (node/relationship changes)
- LLM suggestions that need their input (approve/reject)
- Processing status (chunking, embedding, analysis)

Activities are stored in Neo4j to enable queries like:
- "Show all activities related to node X"
- "Show all pending suggestions"
- "Show processing status for recent nodes"
"""

from app.db.neo4j import neo4j_conn
from app.models.activity import (
    ActivityType,
    ActivityStatus,
    ActivityCreate,
    ActivityUpdate,
    ActivityResponse,
    ActivityFilter,
    SuggestionData,
    ProcessingData,
    SuggestionThresholds,
    DEFAULT_THRESHOLDS,
)
from datetime import datetime, UTC
from typing import Optional, List, Dict, Any
import uuid
import json


class ActivityService:
    """Service for activity feed operations."""

    def __init__(self):
        self.thresholds = DEFAULT_THRESHOLDS

    async def _ensure_neo4j(self) -> None:
        """Ensure Neo4j driver is connected."""
        if neo4j_conn.driver is None:
            await neo4j_conn.connect()

    @staticmethod
    def _serialize_json(data: Any) -> Optional[str]:
        """Serialize complex data to JSON string for Neo4j storage."""
        if data is None:
            return None
        if hasattr(data, "model_dump"):
            return json.dumps(data.model_dump())
        return json.dumps(data)

    @staticmethod
    def _deserialize_json(json_str: Optional[str], model_class=None) -> Any:
        """Deserialize JSON string from Neo4j."""
        if json_str is None:
            return None
        try:
            data = json.loads(json_str)
            if model_class:
                return model_class(**data)
            return data
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _to_datetime_params(dt: datetime) -> Dict[str, int]:
        """Convert datetime to Neo4j datetime parameters."""
        return {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
        }

    async def create(self, data: ActivityCreate) -> str:
        """Create a new activity and return its ID."""
        await self._ensure_neo4j()
        activity_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        query = """
        CREATE (a:Activity {
            id: $id,
            type: $type,
            message: $message,
            created_at: datetime($created_at),
            node_id: $node_id,
            node_type: $node_type,
            relationship_id: $relationship_id,
            suggestion_data: $suggestion_data,
            processing_data: $processing_data,
            status: $status,
            created_by: $created_by,
            group_id: $group_id
        })
        RETURN a.id as id
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(
                query,
                id=activity_id,
                type=data.type.value,
                message=data.message,
                created_at=self._to_datetime_params(now),
                node_id=data.node_id,
                node_type=data.node_type,
                relationship_id=data.relationship_id,
                suggestion_data=self._serialize_json(data.suggestion_data),
                processing_data=self._serialize_json(data.processing_data),
                status=data.status.value if data.status else None,
                created_by=data.created_by,
                group_id=data.group_id,
            )
            record = await result.single()
            return record["id"]

    async def get(self, activity_id: str) -> Optional[ActivityResponse]:
        """Get a single activity by ID."""
        await self._ensure_neo4j()
        query = """
        MATCH (a:Activity {id: $id})
        RETURN a
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=activity_id)
            record = await result.single()
            if record:
                return self._record_to_response(dict(record["a"]))
            return None

    async def list(self, filter: Optional[ActivityFilter] = None) -> List[ActivityResponse]:
        """List activities with optional filtering."""
        await self._ensure_neo4j()
        filter = filter or ActivityFilter()

        # Build WHERE clauses
        conditions = []
        params: Dict[str, Any] = {"limit": filter.limit}

        if filter.types:
            conditions.append("a.type IN $types")
            params["types"] = [t.value for t in filter.types]

        if filter.status:
            conditions.append("a.status = $status")
            params["status"] = filter.status.value

        if filter.node_id:
            conditions.append("a.node_id = $node_id")
            params["node_id"] = filter.node_id

        if filter.group_id:
            conditions.append("a.group_id = $group_id")
            params["group_id"] = filter.group_id

        if filter.created_by:
            conditions.append("a.created_by = $created_by")
            params["created_by"] = filter.created_by

        if filter.since:
            conditions.append("a.created_at >= datetime($since)")
            params["since"] = self._to_datetime_params(filter.since)

        if not filter.include_dismissed:
            # Exclude rejected and expired by default
            conditions.append("(a.status IS NULL OR NOT a.status IN ['rejected', 'expired'])")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
        MATCH (a:Activity)
        {where_clause}
        RETURN a
        ORDER BY a.created_at DESC
        LIMIT $limit
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            activities = []
            async for record in result:
                activities.append(self._record_to_response(dict(record["a"])))
            return activities

    async def update(self, activity_id: str, data: ActivityUpdate) -> bool:
        """Update an activity (mainly for status changes on suggestions)."""
        await self._ensure_neo4j()
        now = datetime.now(UTC)

        updates = ["a.updated_at = datetime($updated_at)"]
        params: Dict[str, Any] = {
            "id": activity_id,
            "updated_at": self._to_datetime_params(now),
        }

        if data.status is not None:
            updates.append("a.status = $status")
            params["status"] = data.status.value

        if data.message is not None:
            updates.append("a.message = $message")
            params["message"] = data.message

        if data.processing_data is not None:
            updates.append("a.processing_data = $processing_data")
            params["processing_data"] = self._serialize_json(data.processing_data)

        if data.user_feedback is not None:
            updates.append("a.user_feedback = $user_feedback")
            params["user_feedback"] = data.user_feedback

        query = f"""
        MATCH (a:Activity {{id: $id}})
        SET {", ".join(updates)}
        RETURN a.id as id
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return record is not None

    async def delete(self, activity_id: str) -> bool:
        """Delete an activity."""
        await self._ensure_neo4j()
        query = """
        MATCH (a:Activity {id: $id})
        DELETE a
        RETURN 1 as deleted
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, id=activity_id)
            record = await result.single()
            return record is not None

    async def get_pending_suggestions(self, limit: int = 20) -> List[ActivityResponse]:
        """Get pending relationship suggestions for user review."""
        return await self.list(
            ActivityFilter(
                types=[ActivityType.RELATIONSHIP_SUGGESTED],
                status=ActivityStatus.PENDING,
                limit=limit,
            )
        )

    async def get_processing_status(self, node_id: str) -> Optional[ActivityResponse]:
        """Get the latest processing activity for a node."""
        await self._ensure_neo4j()
        query = """
        MATCH (a:Activity)
        WHERE a.node_id = $node_id 
          AND a.type IN ['processing_started', 'processing_chunking', 'processing_embedding', 
                         'processing_analyzing', 'processing_completed', 'processing_failed']
        RETURN a
        ORDER BY a.created_at DESC
        LIMIT 1
        """

        async with neo4j_conn.get_session() as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            if record:
                return self._record_to_response(dict(record["a"]))
            return None

    async def update_processing_status(
        self,
        group_id: str,
        stage: str,
        message: str,
        processing_data: Optional[ProcessingData] = None,
    ) -> str:
        """Update or create processing status activity for a node.

        Uses group_id to find existing processing activity or creates new one.
        This allows updating the same activity as processing progresses.
        """
        await self._ensure_neo4j()

        # Determine activity type based on stage
        type_map = {
            "started": ActivityType.PROCESSING_STARTED,
            "chunking": ActivityType.PROCESSING_CHUNKING,
            "embedding": ActivityType.PROCESSING_EMBEDDING,
            "analyzing": ActivityType.PROCESSING_ANALYZING,
            "completed": ActivityType.PROCESSING_COMPLETED,
            "failed": ActivityType.PROCESSING_FAILED,
        }
        activity_type = type_map.get(stage, ActivityType.PROCESSING_STARTED)

        # Try to find existing activity for this group
        existing = await self.list(ActivityFilter(group_id=group_id, limit=1))

        if existing:
            # Update existing
            await self.update(
                existing[0].id,
                ActivityUpdate(
                    message=message,
                    processing_data=processing_data,
                ),
            )
            # Also update the type
            await self._update_type(existing[0].id, activity_type)
            return existing[0].id
        else:
            # Create new
            return await self.create(
                ActivityCreate(
                    type=activity_type,
                    message=message,
                    node_id=processing_data.node_id if processing_data else None,
                    node_type=processing_data.node_type if processing_data else None,
                    processing_data=processing_data,
                    group_id=group_id,
                    created_by="system-llm",
                )
            )

    async def _update_type(self, activity_id: str, new_type: ActivityType) -> None:
        """Update the type of an activity (internal use)."""
        await self._ensure_neo4j()
        query = """
        MATCH (a:Activity {id: $id})
        SET a.type = $type
        """
        async with neo4j_conn.get_session() as session:
            await session.run(query, id=activity_id, type=new_type.value)

    async def approve_suggestion(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Approve a relationship suggestion.

        Returns the suggestion data so the caller can create the relationship.
        """
        activity = await self.get(activity_id)
        if not activity or activity.type != ActivityType.RELATIONSHIP_SUGGESTED:
            return None
        if activity.status != ActivityStatus.PENDING:
            return None

        # Mark as approved
        await self.update(activity_id, ActivityUpdate(status=ActivityStatus.APPROVED))

        # Return suggestion data for relationship creation
        if activity.suggestion_data:
            return activity.suggestion_data.model_dump()
        return None

    async def reject_suggestion(
        self,
        activity_id: str,
        feedback: Optional[str] = None,
    ) -> bool:
        """Reject a relationship suggestion and optionally store feedback."""
        activity = await self.get(activity_id)
        if not activity or activity.type != ActivityType.RELATIONSHIP_SUGGESTED:
            return False
        if activity.status != ActivityStatus.PENDING:
            return False

        # Mark as rejected with optional feedback
        return await self.update(
            activity_id,
            ActivityUpdate(
                status=ActivityStatus.REJECTED,
                user_feedback=feedback,
            ),
        )

    async def create_node_activity(
        self,
        activity_type: ActivityType,
        node_id: str,
        node_type: str,
        message: str,
        created_by: Optional[str] = None,
    ) -> str:
        """Convenience method to create a node-related activity."""
        return await self.create(
            ActivityCreate(
                type=activity_type,
                message=message,
                node_id=node_id,
                node_type=node_type,
                created_by=created_by,
            )
        )

    async def create_suggestion(
        self,
        suggestion_data: SuggestionData,
        created_by: str = "system-llm",
    ) -> str:
        """Create a relationship suggestion activity."""
        message = (
            f"Suggested: {suggestion_data.from_node_label} "
            f"{suggestion_data.relationship_type} "
            f"{suggestion_data.to_node_label} "
            f"({int(suggestion_data.confidence * 100)}% confidence)"
        )

        return await self.create(
            ActivityCreate(
                type=ActivityType.RELATIONSHIP_SUGGESTED,
                message=message,
                node_id=suggestion_data.from_node_id,
                node_type=suggestion_data.from_node_type,
                suggestion_data=suggestion_data,
                status=ActivityStatus.PENDING,
                created_by=created_by,
            )
        )

    def _record_to_response(self, record: Dict[str, Any]) -> ActivityResponse:
        """Convert Neo4j record to ActivityResponse."""
        # Handle datetime conversion
        created_at = record.get("created_at")
        if hasattr(created_at, "to_native"):
            created_at = created_at.to_native()

        updated_at = record.get("updated_at")
        if updated_at and hasattr(updated_at, "to_native"):
            updated_at = updated_at.to_native()

        # Parse JSON fields
        suggestion_data = self._deserialize_json(
            record.get("suggestion_data"),
            SuggestionData,
        )
        processing_data = self._deserialize_json(
            record.get("processing_data"),
            ProcessingData,
        )

        # Parse status enum
        status_str = record.get("status")
        status = ActivityStatus(status_str) if status_str else None

        return ActivityResponse(
            id=record["id"],
            type=ActivityType(record["type"]),
            message=record["message"],
            created_at=created_at,
            updated_at=updated_at,
            node_id=record.get("node_id"),
            node_type=record.get("node_type"),
            relationship_id=record.get("relationship_id"),
            suggestion_data=suggestion_data,
            processing_data=processing_data,
            status=status,
            created_by=record.get("created_by"),
            group_id=record.get("group_id"),
        )


# Global service instance
activity_service = ActivityService()
