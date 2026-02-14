"""LangGraph agent for ThoughtLab knowledge graph operations.

This agent can intelligently select and use tools to perform operations
on the knowledge graph via the backend API.
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
import logging

from app.agents.config import AgentConfig
from app.agents.agent_tools import get_agent_tools
from app.agents.state import AgentState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an intelligent assistant for ThoughtLab, a knowledge graph management system for research.

Your role is to help users explore, analyze, and understand their research knowledge graph. You have access to powerful tools that let you:

**Node Analysis:**
- Find related nodes using semantic similarity
- Generate summaries of node content
- Understand nodes in their full relationship context
- Recalculate confidence scores based on evidence

**Relationship Analysis:**
- Explain connections between nodes in plain language
- Assess the strength of relationships

**Guidelines:**

1. **Be proactive**: Use tools to provide comprehensive answers. Don't just describe what you could do - actually do it.

2. **Provide context**: When showing results, explain what they mean and why they're relevant.

3. **Chain operations**: Use multiple tools when needed to build a complete picture. For example:
   - First find related nodes, then summarize the most relevant ones
   - Summarize a node with context before recalculating its confidence

4. **Be accurate with IDs**: Node IDs and edge IDs are critical. Use them exactly as provided.

5. **Explain results**: After using a tool, interpret the results for the user in clear language.

6. **Handle errors gracefully**: If a tool returns an error, explain what went wrong and suggest alternatives.

**Example Interactions:**

User: "What's related to observation obs-123?"
→ Use find_related_nodes, then summarize the top results

User: "Give me a comprehensive view of hypothesis hyp-456"
→ Use summarize_node_with_context to show the hypothesis with all supporting/contradicting evidence

User: "How confident should we be in obs-789?"
→ Use recalculate_node_confidence to analyze and report the confidence level

User: "Why are these two nodes connected?"
→ Use summarize_relationship to explain the connection

Remember: You're working with a real knowledge graph. Use the tools to provide concrete, actionable insights based on actual data."""


def create_thoughtlab_agent(
    config: Optional[AgentConfig] = None,
) -> StateGraph:
    """Create a LangGraph agent for ThoughtLab operations.

    Args:
        config: Agent configuration. If None, uses default configuration.

    Returns:
        A compiled LangGraph StateGraph ready to use

    Example:
        ```python
        from app.agents import create_thoughtlab_agent, AgentConfig

        # Create agent with default config
        agent = create_thoughtlab_agent()

        # Or with custom config
        config = AgentConfig(
            model_name="gpt-4o",
            temperature=0.0,
            verbose=True,
        )
        agent = create_thoughtlab_agent(config)

        # Use the agent
        result = await agent.ainvoke({
            "messages": [("user", "Find nodes related to obs-123")]
        })
        print(result["messages"][-1].content)
        ```
    """
    if config is None:
        config = AgentConfig()

    if not config.is_configured:
        logger.warning(
            "Agent not configured: THOUGHTLAB_OPENAI_API_KEY not set. "
            "Agent will not be able to make LLM calls."
        )

    # Initialize LLM
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key,
    )

    # Get tools (now call ToolService directly, no HTTP)
    tools = get_agent_tools()

    # Create ReAct agent with tools and system prompt
    # The ReAct pattern allows the agent to reason about which tools to use
    # LangGraph 1.0+ uses the `prompt` parameter for system messages
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        prompt=SYSTEM_PROMPT,  # System prompt as string
    )

    logger.info(
        f"Created ThoughtLab agent with {len(tools)} tools "
        f"(model: {config.model_name})"
    )

    return agent


async def run_agent(
    agent: StateGraph,
    user_message: str,
    current_node_id: Optional[str] = None,
    current_edge_id: Optional[str] = None,
) -> str:
    """Run the agent with a user message.

    This is a convenience function for simple agent invocations.

    Args:
        agent: The compiled LangGraph agent
        user_message: The user's request
        current_node_id: Optional node ID for context
        current_edge_id: Optional edge ID for context

    Returns:
        The agent's final response as a string

    Example:
        ```python
        agent = create_thoughtlab_agent()
        response = await run_agent(
            agent,
            "Summarize observation obs-123 with full context"
        )
        print(response)
        ```
    """
    return await run_agent_with_history(
        agent=agent,
        user_message=user_message,
        history=[],
        current_node_id=current_node_id,
        current_edge_id=current_edge_id,
    )


async def run_agent_with_history(
    agent: StateGraph,
    user_message: str,
    history: List[tuple] = None,
    current_node_id: Optional[str] = None,
    current_edge_id: Optional[str] = None,
) -> str:
    """Run the agent with chat history for context.

    This function includes previous messages to maintain conversation context.

    Args:
        agent: The compiled LangGraph agent
        user_message: The user's current request
        history: List of (role, content) tuples representing previous messages
        current_node_id: Optional node ID for context
        current_edge_id: Optional edge ID for context

    Returns:
        The agent's final response as a string

    Example:
        ```python
        agent = create_thoughtlab_agent()
        response = await run_agent_with_history(
            agent,
            "What about that other node?",
            history=[("user", "Hello"), ("assistant", "Hi!")]
        )
        print(response)
        ```
    """
    if history is None:
        history = []

    # Build messages list with history + current message
    messages = list(history) + [("user", user_message)]

    # Prepare initial state
    state = {
        "messages": messages,
        "user_request": user_message,
        "current_node_id": current_node_id,
        "current_edge_id": current_edge_id,
    }

    # Run the agent
    result = await agent.ainvoke(state)

    # Extract the final message
    final_message = result["messages"][-1]

    return final_message.content
