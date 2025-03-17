"""
This module defines the configuration models for a graph-based remote agent system.

Classes:
    RemoteAgentConfig: Configuration for the remote agent, including URL, ID, model, and metadata.
    GraphConfig: Configuration for the graph execution, including thread ID, REST timeout, and remote agent details.

Attributes:
    RemoteAgentConfig.url (str): URL of the remote agent service. Defaults to local server.
    RemoteAgentConfig.id (str): ID of the remote agent.
    RemoteAgentConfig.model (str): Model used by the remote agent.
    RemoteAgentConfig.metadata (Dict[str, Any]): Metadata for the remote agent.
    GraphConfig.thread_id (Optional[str]): Optional unique identifier for the execution thread.
    GraphConfig.rest_timeout (int): Timeout (in seconds) for REST API requests.
    GraphConfig.remote_agent (RemoteAgentConfig): Details of the remote agent including URL, ID, model, and metadata.
"""

import uuid

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RemoteAgentConfig(BaseModel):
    """Configuration for the remote agent."""

    url: str = Field(
        "http://127.0.0.1:8123/api/v1/runs",
        description="URL of the remote agent service. Defaults to local server.",
    )
    id: str = Field("remote_agent", description="ID of the remote agent.")
    model: str = Field("gpt-4o", description="Model used by the remote agent.")
    metadata: Dict[str, Any] = Field({}, description="Metadata for the remote agent.")


class GraphConfig(BaseModel):
    """Configuration for the graph execution, including remote agent details."""

    thread_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional unique identifier for the execution thread.",
    )

    rest_timeout: int = Field(
        30, description="Timeout (in seconds) for REST API requests."
    )

    remote_agent: RemoteAgentConfig = Field(
        ...,
        description="Details of the remote agent including URL, ID, model, and metadata.",
    )
