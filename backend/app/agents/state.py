"""Agent state management for LangGraph workflows."""

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State for ThoughtLab LangGraph agent.

    This state tracks the conversation history and context
    throughout the agent's reasoning process.

    Note: LangGraph 1.0+ requires certain fields for react agents.
    """

    # Required by LangGraph's create_react_agent
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: int  # Tracks iterations remaining (required by LangGraph)

    # Optional: Track which nodes/edges the agent is working with
    current_node_id: Optional[str]
    current_edge_id: Optional[str]

    # Optional: Track the user's original request
    user_request: Optional[str]
