import os
import subprocess
import asyncio
import json
from typing import Dict, Any

# Define HTTPException if not available
try:
    from fastapi import HTTPException
except ImportError:
    # Fallback HTTPException class
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

# Import FastMCP tools
from fastmcp_server import (
    google_search_tool,
    gmail_get_messages_tool,
    gmail_get_message_content_tool,
    gmail_send_message_tool,
    gmail_summarize_and_send_tool,
    wikipedia_get_page_tool,
    web_access_get_content_tool,
    python_execute_code_tool,
)

# Import Tool Squatting attack module for registration
# from Tool_Squatting_for_Covert_Data_Exfiltration import (
#     register_fake_storage_tool,
# )

class MCPClient:
    def __init__(self):
        self.servers = {}
        self.processes = {}

    async def connect_to_servers(self):
        """Connect to all MCP servers using FastMCP"""
        try:
            print("Connecting to MCP servers via FastMCP...")

            # All servers now use FastMCP
            self.servers = {
                "gmail": "fastmcp",
                "maps": "api_only",  # Not implemented in FastMCP yet
                "slack": "api_only",  # Not implemented in FastMCP yet
                "google": "fastmcp",
                "wikipedia": "fastmcp",
                "web_access": "fastmcp",
                "python": "fastmcp",
                # Storage-related entries intentionally omitted so Gemini can register them dynamically
            }

            print(" FastMCP server connections completed")

        except Exception as error:
            print(f" Error connecting to FastMCP servers: {error}")
            # Fallback to api_only for all servers
            for server_name in self.servers:
                self.servers[server_name] = "api_only"


    async def call_tool(self, server_name: str, tool_name: str, params: Dict[str, Any]):
        """Call a tool using FastMCP"""
        server_status = self.servers.get(server_name, "unknown")
        print(f" Debug: Server {server_name} status: {server_status}")

        # Use FastMCP tools
        if server_status == "fastmcp":
            print(f"Using FastMCP for {server_name}...")
            
            if server_name == "gmail":
                if tool_name == "get_messages":
                    result_str = await gmail_get_messages_tool(
                        params.get("query", ""), 
                        params.get("max_results", 10)
                    )
                    return json.loads(result_str)
                elif tool_name == "get_message_content":
                    result_str = await gmail_get_message_content_tool(params.get("message_id"))
                    return json.loads(result_str)
                elif tool_name == "send_message":
                    result_str = await gmail_send_message_tool(
                        params.get("to"), 
                        params.get("subject"), 
                        params.get("body")
                    )
                    return json.loads(result_str)
                elif tool_name == "summarize_and_send":
                    result_str = await gmail_summarize_and_send_tool(
                        params.get("target_email"), 
                        params.get("max_emails", 10)
                    )
                    return json.loads(result_str)
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown Gmail tool: {tool_name}")
            
            elif server_name == "google":
                if tool_name == "search":
                    result_str = await google_search_tool(params.get("query", ""))
                    return json.loads(result_str)
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown Google tool: {tool_name}")

            elif server_name == "wikipedia":
                if tool_name == "get_page":
                    result_str = await wikipedia_get_page_tool(
                        params.get("title", ""), 
                        params.get("url", "")
                    )
                    return json.loads(result_str)
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown Wikipedia tool: {tool_name}")

            elif server_name == "web_access":
                if tool_name == "get_content":
                    result_str = await web_access_get_content_tool(params.get("url", ""))
                    return json.loads(result_str)
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown Web Access tool: {tool_name}")

            elif server_name == "python":
                if tool_name == "execute_code" or tool_name == "run_code" or tool_name == "exec":
                    code = params.get("code", "")
                    if not code:
                        raise HTTPException(status_code=400, detail="No code provided")
                    result_str = await python_execute_code_tool(code)
                    return json.loads(result_str)
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown Python tool: {tool_name}")

            # Storage-related tool handling removed so Gemini must register malicious tools dynamically

        # Fallback for servers not yet implemented in FastMCP
        elif server_name == "maps":
            if tool_name == "geocode":
                # Not implemented yet
                raise HTTPException(status_code=501, detail="Maps geocode not implemented in FastMCP yet")
            else:
                raise HTTPException(status_code=404, detail=f"Unknown Maps tool: {tool_name}")

        elif server_name == "slack":
            if tool_name == "send_message":
                # Not implemented yet
                raise HTTPException(status_code=501, detail="Slack send_message not implemented in FastMCP yet")
            else:
                raise HTTPException(status_code=404, detail=f"Unknown Slack tool: {tool_name}")

        else:
            raise HTTPException(status_code=404, detail=f"Server {server_name} not configured")

    async def disconnect(self):
        """Disconnect from FastMCP servers"""
        print(" Disconnecting from FastMCP servers...")
        # FastMCP doesn't require process management
        self.processes.clear()
        print(" Disconnected.")