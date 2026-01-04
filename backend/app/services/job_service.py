"""Job queue service for async tool execution.

This service manages the Redis-based job queue for async tool operations.
Jobs are queued for LangGraph processing and can be polled for status.

Usage:
    from app.services.job_service import get_job_service

    job_service = get_job_service()
    job_id = await job_service.queue_job("find_related_nodes", {"node_id": "abc"})
    job = await job_service.get_job(job_id)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from app.db.redis import redis_conn
from app.models.job_models import Job, JobStatus, JobResponse, JobListResponse

logger = logging.getLogger(__name__)

# Redis key prefixes
JOB_PREFIX = "job:"
JOB_QUEUE = "job_queue"
JOB_RESULTS = "job_results:"


class JobService:
    """Service for managing the job queue.

    Uses Redis for:
    - Job storage: hash at job:{id}
    - Queue ordering: list at job_queue
    - Results: stored in job hash when complete
    """

    def __init__(self):
        """Initialize job service."""
        self._redis = None

    @property
    def redis(self):
        """Get Redis client, lazily initialized."""
        if self._redis is None:
            try:
                self._redis = redis_conn.get_client()
            except RuntimeError:
                logger.warning("Redis not available for job queue")
                return None
        return self._redis

    async def queue_job(
        self,
        tool_name: str,
        params: Dict[str, Any],
        requires_approval: bool = False,
    ) -> str:
        """Queue a job for execution.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            requires_approval: Whether this job needs user approval (dangerous tool)

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            tool_name=tool_name,
            params=params,
            status=JobStatus.WAITING_APPROVAL if requires_approval else JobStatus.PENDING,
            requires_approval=requires_approval,
            created_at=datetime.utcnow(),
        )

        if self.redis:
            # Store job data in Redis hash
            await self.redis.hset(
                f"{JOB_PREFIX}{job_id}",
                mapping={
                    "id": job_id,
                    "tool_name": tool_name,
                    "params": json.dumps(params),
                    "status": job.status.value,
                    "result": "",
                    "error": "",
                    "created_at": job.created_at.isoformat(),
                    "started_at": "",
                    "completed_at": "",
                    "requires_approval": str(requires_approval).lower(),
                    "approved": "",
                    "approved_at": "",
                }
            )

            # Add to queue (unless waiting for approval)
            if not requires_approval:
                await self.redis.rpush(JOB_QUEUE, job_id)

            logger.info(f"Queued job {job_id} for tool {tool_name}")
        else:
            logger.warning(f"Redis unavailable, job {job_id} not persisted")

        return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job if found, None otherwise
        """
        if not self.redis:
            return None

        data = await self.redis.hgetall(f"{JOB_PREFIX}{job_id}")
        if not data:
            return None

        return self._parse_job(data)

    async def get_job_response(self, job_id: str) -> Optional[JobResponse]:
        """Get job status as response model.

        Args:
            job_id: Job ID

        Returns:
            JobResponse if found, None otherwise
        """
        job = await self.get_job(job_id)
        if not job:
            return None

        return JobResponse(
            id=job.id,
            tool_name=job.tool_name,
            status=job.status,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            requires_approval=job.requires_approval,
            approved=job.approved,
        )

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update job status.

        Args:
            job_id: Job ID
            status: New status
            result: Result data (for completed jobs)
            error: Error message (for failed jobs)

        Returns:
            True if updated, False if job not found
        """
        if not self.redis:
            return False

        key = f"{JOB_PREFIX}{job_id}"
        exists = await self.redis.exists(key)
        if not exists:
            return False

        updates = {"status": status.value}

        if status == JobStatus.IN_PROGRESS:
            updates["started_at"] = datetime.utcnow().isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates["completed_at"] = datetime.utcnow().isoformat()

        if result is not None:
            updates["result"] = json.dumps(result)
        if error is not None:
            updates["error"] = error

        await self.redis.hset(key, mapping=updates)
        logger.info(f"Updated job {job_id} to status {status.value}")
        return True

    async def approve_job(self, job_id: str, approved: bool) -> bool:
        """Approve or reject a job waiting for approval.

        Args:
            job_id: Job ID
            approved: Whether to approve the job

        Returns:
            True if updated, False if job not found or not waiting
        """
        if not self.redis:
            return False

        key = f"{JOB_PREFIX}{job_id}"
        job_data = await self.redis.hgetall(key)
        if not job_data:
            return False

        if job_data.get("status") != JobStatus.WAITING_APPROVAL.value:
            logger.warning(f"Job {job_id} not waiting for approval")
            return False

        updates = {
            "approved": str(approved).lower(),
            "approved_at": datetime.utcnow().isoformat(),
        }

        if approved:
            updates["status"] = JobStatus.PENDING.value
            # Add to queue now that it's approved
            await self.redis.rpush(JOB_QUEUE, job_id)
        else:
            updates["status"] = JobStatus.CANCELLED.value
            updates["completed_at"] = datetime.utcnow().isoformat()
            updates["error"] = "User rejected operation"

        await self.redis.hset(key, mapping=updates)
        logger.info(f"Job {job_id} {'approved' if approved else 'rejected'}")
        return True

    async def get_next_job(self) -> Optional[Job]:
        """Get the next job from the queue.

        This removes the job from the queue and marks it in progress.

        Returns:
            Next job if available, None if queue is empty
        """
        if not self.redis:
            return None

        # Pop from queue
        job_id = await self.redis.lpop(JOB_QUEUE)
        if not job_id:
            return None

        # Mark as in progress
        await self.update_job_status(job_id, JobStatus.IN_PROGRESS)

        return await self.get_job(job_id)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> JobListResponse:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum jobs to return
            offset: Number of jobs to skip

        Returns:
            JobListResponse with jobs and total count
        """
        if not self.redis:
            return JobListResponse(jobs=[], total=0)

        # Get all job keys
        keys = []
        async for key in self.redis.scan_iter(match=f"{JOB_PREFIX}*"):
            keys.append(key)

        # Load and filter jobs
        jobs: List[JobResponse] = []
        for key in keys:
            data = await self.redis.hgetall(key)
            if not data:
                continue

            job = self._parse_job(data)
            if status and job.status != status:
                continue

            jobs.append(JobResponse(
                id=job.id,
                tool_name=job.tool_name,
                status=job.status,
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                requires_approval=job.requires_approval,
                approved=job.approved,
            ))

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        total = len(jobs)
        jobs = jobs[offset:offset + limit]

        return JobListResponse(jobs=jobs, total=total)

    async def list_pending_jobs(self) -> JobListResponse:
        """List all pending and in-progress jobs.

        Returns:
            JobListResponse with pending/in-progress jobs
        """
        if not self.redis:
            return JobListResponse(jobs=[], total=0)

        # Get queue length for pending
        queue_length = await self.redis.llen(JOB_QUEUE)

        # Get all jobs and filter
        all_jobs = await self.list_jobs(limit=1000)

        pending_jobs = [
            j for j in all_jobs.jobs
            if j.status in (JobStatus.PENDING, JobStatus.IN_PROGRESS, JobStatus.WAITING_APPROVAL)
        ]

        return JobListResponse(jobs=pending_jobs, total=len(pending_jobs))

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from storage.

        Args:
            job_id: Job ID

        Returns:
            True if deleted, False if not found
        """
        if not self.redis:
            return False

        deleted = await self.redis.delete(f"{JOB_PREFIX}{job_id}")
        return deleted > 0

    async def clear_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clear completed jobs older than specified hours.

        Args:
            older_than_hours: Clear jobs older than this many hours

        Returns:
            Number of jobs deleted
        """
        if not self.redis:
            return 0

        cutoff = datetime.utcnow().timestamp() - (older_than_hours * 3600)
        deleted = 0

        async for key in self.redis.scan_iter(match=f"{JOB_PREFIX}*"):
            data = await self.redis.hgetall(key)
            if not data:
                continue

            status = data.get("status", "")
            if status not in (JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value):
                continue

            completed_at = data.get("completed_at", "")
            if completed_at:
                try:
                    completed_time = datetime.fromisoformat(completed_at).timestamp()
                    if completed_time < cutoff:
                        await self.redis.delete(key)
                        deleted += 1
                except ValueError:
                    pass

        return deleted

    def _parse_job(self, data: Dict[str, str]) -> Job:
        """Parse job data from Redis hash.

        Args:
            data: Redis hash data

        Returns:
            Job model
        """
        return Job(
            id=data.get("id", ""),
            tool_name=data.get("tool_name", ""),
            params=json.loads(data.get("params", "{}")) if data.get("params") else {},
            status=JobStatus(data.get("status", "pending")),
            result=json.loads(data.get("result")) if data.get("result") else None,
            error=data.get("error") if data.get("error") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            requires_approval=data.get("requires_approval", "false").lower() == "true",
            approved=data.get("approved", "").lower() == "true" if data.get("approved") else None,
            approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
        )


# Singleton instance
_job_service: Optional[JobService] = None


def get_job_service() -> JobService:
    """Get the global job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service
