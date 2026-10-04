from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ReviewStatus(str, Enum):
    proposed = "proposed"
    accepted = "accepted"
    rejected = "rejected"
    edited = "edited"


class EntityType(str, Enum):
    client = "client"
    engagement = "engagement"
    person = "person"
    artifact = "artifact"
    requirement = "requirement"
    finding = "finding"
    deliverable = "deliverable"
    theme = "theme"


# --- Pydantic API / extraction schemas (bilingual-aware) ---


class ProvenanceFields(BaseModel):
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_artifact_id: Optional[str] = None
    span_ref: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.proposed
    language: Optional[str] = Field(default=None, description="en | de | mixed")


class ClientRecord(ProvenanceFields):
    name: str
    aliases: list[str] = Field(default_factory=list)
    industry: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class EngagementRecord(ProvenanceFields):
    title: str
    client_name: Optional[str] = None
    type: Optional[str] = None
    phase: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    status: Optional[str] = None
    commercial_model: Optional[str] = None


class PersonRecord(ProvenanceFields):
    name: str
    role: Optional[str] = None
    org: Optional[str] = None
    email: Optional[str] = None
    engagement_titles: list[str] = Field(default_factory=list)


class ArtifactRecord(ProvenanceFields):
    title: str
    type: Optional[str] = None
    path: Optional[str] = None
    content_hash: Optional[str] = None
    engagement_title: Optional[str] = None
    authored_at: Optional[str] = None
    ocr_used: bool = False


class RequirementRecord(ProvenanceFields):
    statement: str
    priority: Optional[str] = None
    status: Optional[str] = None


class FindingRecord(ProvenanceFields):
    statement: str
    severity: Optional[str] = None
    theme: Optional[str] = None
    status: Optional[str] = None


class DeliverableRecord(ProvenanceFields):
    name: str
    type: Optional[str] = None
    due_date: Optional[str] = None
    engagement_title: Optional[str] = None


class ThemeRecord(ProvenanceFields):
    label: str
    stability_score: float = 0.0
    member_chunk_ids: list[str] = Field(default_factory=list)
    persistence_summary: Optional[str] = None


class ExtractionBundle(BaseModel):
    clients: list[ClientRecord] = Field(default_factory=list)
    engagements: list[EngagementRecord] = Field(default_factory=list)
    people: list[PersonRecord] = Field(default_factory=list)
    requirements: list[RequirementRecord] = Field(default_factory=list)
    findings: list[FindingRecord] = Field(default_factory=list)
    deliverables: list[DeliverableRecord] = Field(default_factory=list)


class ReviewUpdate(BaseModel):
    review_status: ReviewStatus
    edits: dict[str, Any] = Field(default_factory=dict)


# --- SQLAlchemy persistence ---


class Base(DeclarativeBase):
    pass


class ProposedEntity(Base):
    __tablename__ = "proposed_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512))
    payload: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_artifact_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    span_ref: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default=ReviewStatus.proposed.value, index=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ArtifactRow(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    path: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str] = mapped_column(String(128))
    mime_hint: Mapped[str] = mapped_column(String(64))
    text_preview: Mapped[str] = mapped_column(Text, default="")
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_dir: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    llm_mode: Mapped[str] = mapped_column(String(32), default="mock")
    theme_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
