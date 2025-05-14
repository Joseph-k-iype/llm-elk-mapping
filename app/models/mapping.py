"""
Models for mapping functionalities.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.base import BaseResponse

class BusinessTerm(BaseModel):
    """
    Business term model.
    """
    id: str
    pbt_name: str = Field(..., alias="PBT_NAME")
    pbt_definition: str = Field(..., alias="PBT_DEFINITION")
    cdm: Optional[str] = Field(None, alias="CDM")

    class Config:
        populate_by_name = True

class MappingRequest(BaseModel):
    """
    Request model for business term mapping.
    """
    name: str
    description: str
    example: Optional[str] = None
    cdm: Optional[str] = None
    process_name: Optional[str] = None
    process_description: Optional[str] = None

class MappingResult(BaseModel):
    """
    Result model for a single mapping match.
    """
    term_id: str
    term_name: str
    similarity_score: float
    confidence: float
    mapping_type: str  # semantic, BM25, keyword, agent
    matched_attributes: List[str]

class MappingResponse(BaseResponse):
    """
    Response model for mapping requests.
    """
    results: List[MappingResult] = []
    query: Optional[MappingRequest] = None
    timestamp: datetime = Field(default_factory=datetime.now)