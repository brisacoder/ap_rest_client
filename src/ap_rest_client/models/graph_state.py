
"""
This module defines the GraphState model, which represents the state of a Langgraph graph.
It includes a list of messages exchanged within the session and an optional error message
in case of exceptions during execution.

Classes:
    GraphState: A Pydantic model representing the state of the graph.
"""

from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """Represents the state of the graph, containing messages and an optional error message."""

    messages: Annotated[List[BaseMessage], add_messages] = Field(
        ..., description="List of messages exchanged within the graph session."
    )

    exception_msg: Optional[str] = Field(
        None,
        description="Optional error message in case of exceptions during execution.",
    )
