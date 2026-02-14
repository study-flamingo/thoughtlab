"""Chat API routes for AI assistant.

Provides endpoints for natural language interaction with the
ThoughtLab knowledge graph via the LangGraph agent.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.agents import create_thoughtlab_agent, run_agent_with_history
from app.agents.config import AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """A single message in the chat history."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    history: List[ChatMessage] = []
    session_id: Optional[str] = None
    current_node_id: Optional[str] = None
    current_edge_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    session_id: Optional[str] = None
    actions: list = []


# Global agent instance (lazy loaded)
_agent = None


def get_agent():
    """Get or create the ThoughtLab agent."""
    global _agent
    if _agent is None:
        config = AgentConfig()
        if not config.is_configured:
            logger.warning("Agent not configured - OpenAI API key not set")
        _agent = create_thoughtlab_agent(config)
    return _agent


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the AI response.

    This endpoint allows users to interact with the knowledge graph
    using natural language. The AI can:
    - Answer questions about nodes and relationships
    - Create new nodes from URLs or descriptions
    - Find related nodes and suggest connections
    - Summarize and analyze content

    Example:
        ```json
        {
            "message": "Create a source from https://example.com/article",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ],
            "current_node_id": "optional-context-node"
        }
        ```
    """
    try:
        agent = get_agent()

        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="AI assistant not configured. Please set OPENAI_API_KEY."
            )

        # Convert history to LangChain message format
        history = [(msg.role, msg.content) for msg in request.history]

        # Run the agent with history
        response_text = await run_agent_with_history(
            agent=agent,
            user_message=request.message,
            history=history,
            current_node_id=request.current_node_id,
            current_edge_id=request.current_edge_id,
        )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id or "default",
            actions=[],  # TODO: Parse actions from response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """Check if the chat/AI service is healthy."""
    config = AgentConfig()
    return {
        "configured": config.is_configured,
        "model": config.model_name,
        "agent_ready": _agent is not None,
    }
