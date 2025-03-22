"""
This module defines the GraphState model, which represents the state of a Langgraph graph.
It includes a list of messages exchanged within the session and an optional error message
in case of exceptions during execution.

Classes:
    GraphState: A Pydantic model representing the state of the graph.
"""

from typing import Any, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ThreadState(BaseModel):
    """Represents input details for the graph."""

    messages: List[BaseMessage] = Field(
        ..., description="List of messages exchanged within the graph session."
    )

    extended_info: Any = Field(
        None,
        description="Object added to each remote call to provide additional input details.",
    )


class GraphState(BaseModel):
    """Represents the state of the graph, containing messages and an optional error message."""

    exception_msg: Optional[str] = Field(
        None,
        description="Optional error message in case of exceptions during execution.",
    )

    thread_state: ThreadState = Field(
        ...,
        description="Messages and additional input details for the graph."
    )
