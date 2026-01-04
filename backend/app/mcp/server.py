"""ThoughtLab MCP Server implementation using FastMCP.

This server exposes ThoughtLab's knowledge graph operations as MCP tools,
allowing Claude Desktop and other MCP clients to interact with the graph.

All tools call ToolService directly (in-process), providing:
- Sync tools: Immediate execution, returns result
- Async tools: Queues job, client polls with check_job_status
- Dangerous tools: Gated by THOUGHTLAB_MCP_ADMIN_MODE env var

Usage:
    # Mount in FastAPI app at /mcp
    from app.mcp import create_mcp_server

    mcp = create_mcp_server()
    app.mount("/mcp", mcp.http_app())

    # Or run standalone
    mcp = create_mcp_server()
    mcp.run()
"""

import os
import logging

from fastmcp import FastMCP

from app.mcp.mcp_tools import register_mcp_tools, ADMIN_MODE

logger = logging.getLogger(__name__)


def create_mcp_server(server_name: str = "ThoughtLab") -> FastMCP:
    """Create and configure the ThoughtLab MCP server.

    The server is configured with all enabled tools based on:
    - Tool definitions from app.tools
    - THOUGHTLAB_MCP_ADMIN_MODE env var for dangerous tool access

    Args:
        server_name: Name for the MCP server

    Returns:
        Configured FastMCP server instance

    Example:
        # Mount in FastAPI
        mcp = create_mcp_server()
        app.mount("/mcp", mcp.http_app())

        # Run standalone
        mcp = create_mcp_server()
        mcp.run()
    """
    mcp = FastMCP(server_name)

    # Register tools from tool definitions
    register_mcp_tools(mcp)

    # Add health check tool
    @mcp.tool(name="check_api_health", description="Check if the ThoughtLab backend is healthy and accessible.")
    async def check_api_health() -> str:
        """Check backend health status."""
        from app.db.neo4j import neo4j_conn
        from app.db.redis import redis_conn
        from app.ai.config import AIConfig

        output = ["ThoughtLab Health Check:\n\n"]

        # Check Neo4j
        try:
            if neo4j_conn.driver is not None:
                await neo4j_conn.driver.verify_connectivity()
                output.append("Neo4j: healthy\n")
            else:
                output.append("Neo4j: not connected\n")
        except Exception as e:
            output.append(f"Neo4j: unhealthy ({e})\n")

        # Check Redis
        try:
            if redis_conn.client is not None:
                await redis_conn.get_client().ping()
                output.append("Redis: healthy\n")
            else:
                output.append("Redis: not configured\n")
        except Exception as e:
            output.append(f"Redis: unavailable ({e})\n")

        # Check AI config
        ai_config = AIConfig()
        output.append(f"AI Configured: {'Yes' if ai_config.openai_api_key else 'No'}\n")
        output.append(f"LLM Model: {ai_config.model_name}\n")
        output.append(f"Admin Mode: {'Enabled' if ADMIN_MODE else 'Disabled'}\n")

        return "".join(output)

    logger.info(
        f"Created ThoughtLab MCP server '{server_name}' "
        f"(admin_mode={ADMIN_MODE})"
    )

    return mcp


# For standalone execution
if __name__ == "__main__":
    mcp = create_mcp_server()
    mcp.run()
