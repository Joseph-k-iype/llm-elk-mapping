"""
Models for mapping job tracking with mandatory ID fields.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from app.models.base import BaseResponse
from app.models.mapping import MappingRequest, MappingResult


class JobStatus(str, Enum):
    """Enum for job status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MappingJob(BaseModel):
    """
    Model for tracking mapping jobs.
    """
    id: str  # Now required - no default factory
    process_id: str  # Now required - no default factory
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    request: MappingRequest
    results: Optional[List[MappingResult]] = None
    error: Optional[str] = None
    
    def update_status(self, status: JobStatus):
        """Update job status and updated_at timestamp."""
        self.status = status
        self.updated_at = datetime.now()
        if status == JobStatus.COMPLETED:
            self.completed_at = datetime.now()


class MappingJobResponse(BaseResponse):
    """
    Response model for mapping job endpoints.
    """
    job: Optional[MappingJob] = None


class MappingJobsResponse(BaseResponse):
    """
    Response model for listing mapping jobs.
    """
    jobs: List[MappingJob] = []
    total: int = 0
    page: int = 1
    page_size: int = 10


class JobStatusUpdateRequest(BaseModel):
    """
    Request model for updating job status.
    """
    status: JobStatus
    results: Optional[List[MappingResult]] = None
    error: Optional[str] = None


class MappingJobRequest(BaseModel):
    """
    Request model for creating a new mapping job,
    combining the mapping request with job identifiers.
    """
    id: str
    process_id: str
    name: str
    description: str
    example: Optional[str] = None
    cdm: Optional[str] = None
    process_name: Optional[str] = None
    process_description: Optional[str] = None
    
    def to_mapping_request(self) -> MappingRequest:
        """Convert to a MappingRequest object."""
        return MappingRequest(
            name=self.name,
            description=self.description,
            example=self.example,
            cdm=self.cdm,
            process_name=self.process_name,
            process_description=self.process_description
        )