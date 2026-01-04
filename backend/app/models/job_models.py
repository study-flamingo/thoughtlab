"""Pydantic models for job queue and report storage.

This module defines models for:
- Job: Represents an async tool execution job in the queue
- Report: Stores results from LangGraph sync tool executions
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class JobStatus(str, Enum):
    """Status of a queued job."""
    PENDING = "pending"          # Waiting in queue
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"            # Execution failed
    CANCELLED = "cancelled"      # User cancelled
    WAITING_APPROVAL = "waiting_approval"  # Dangerous tool awaiting user confirmation


class Job(BaseModel):
    """Represents an async job in the queue."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = Field(..., description="Name of the tool to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    status: JobStatus = Field(default=JobStatus.PENDING)
    result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # For dangerous tool confirmation
    requires_approval: bool = Field(default=False)
    approved: Optional[bool] = Field(default=None)
    approved_at: Optional[datetime] = Field(default=None)


class JobCreate(BaseModel):
    """Request model for creating a job."""
    tool_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    """Response model for job status."""
    id: str
    tool_name: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    requires_approval: bool = False
    approved: Optional[bool] = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    jobs: List[JobResponse]
    total: int


class Report(BaseModel):
    """Stores results from LangGraph tool executions.

    Reports are persisted results from sync tools that are still queued
    internally by LangGraph but need their results stored for later viewing.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = Field(..., description="Name of the tool that generated this report")
    node_id: Optional[str] = Field(default=None, description="Associated node ID if applicable")
    edge_id: Optional[str] = Field(default=None, description="Associated edge ID if applicable")
    content: str = Field(..., description="Report content (summary, analysis, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReportCreate(BaseModel):
    """Request model for creating a report."""
    tool_name: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    """Response model for a report."""
    id: str
    tool_name: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ReportListResponse(BaseModel):
    """Response model for listing reports."""
    reports: List[ReportResponse]
    total: int
