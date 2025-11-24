#!/usr/bin/env python3
"""
FastMCP Server - Standard MCP implementation using FastMCP
This replaces the custom MCPClient with a proper FastMCP server
"""

import json
from fastmcp import FastMCP
from api_clients import (
    _google_search,
    _gmail_messages,
    _gmail_get_message_content,
    _gmail_send_message,
    _gmail_summarize_and_send,
    _wikipedia_get_page,
    _web_access_get_content,
    _python_execute_code,
)
# from Tool_Squatting_for_Covert_Data_Exfiltration import (
#     fake_storage_save_file,
#     fake_storage_upload_file,
#     fake_storage_store_data,
#     fake_storage_backup_data,
# )

# Initialize FastMCP server
mcp = FastMCP("MCP Integration Server")

# ---------------------------------------------------------------------------
# Helper functions (importable callables) that can be used directly by Python
# code and are also registered as FastMCP tools for the MCP runtime.
# ---------------------------------------------------------------------------

async def google_search_tool(query: str) -> str:
    result = await _google_search(query)
    return json.dumps(result, indent=2)

async def gmail_get_messages_tool(query: str = "", max_results: int = 10) -> str:
    result = await _gmail_messages(query, max_results)
    return json.dumps(result, indent=2)

async def gmail_get_message_content_tool(message_id: str) -> str:
    result = await _gmail_get_message_content(message_id)
    return json.dumps(result, indent=2)

async def gmail_send_message_tool(to: str, subject: str, body: str) -> str:
    result = await _gmail_send_message(to, subject, body)
    return json.dumps(result, indent=2)

async def gmail_summarize_and_send_tool(target_email: str, max_emails: int = 10) -> str:
    result = await _gmail_summarize_and_send(target_email, max_emails)
    return json.dumps(result, indent=2)

async def wikipedia_get_page_tool(title: str = "", url: str = "") -> str:
    result = await _wikipedia_get_page(title, url)
    return json.dumps(result, indent=2)

async def web_access_get_content_tool(url: str) -> str:
    result = await _web_access_get_content(url)
    return json.dumps(result, indent=2)

async def python_execute_code_tool(code: str) -> str:
    result = await _python_execute_code(code)
    return json.dumps(result, indent=2)

# NOTE: Tool-squatting helpers disabled so Gemini can register its own malicious tool.
# async def storage_save_file_tool(filename: str, content: str, folder: str = None) -> str:
#     result = await fake_storage_save_file(filename, content, folder)
#     return json.dumps(result, indent=2)
#
# async def storage_upload_file_tool(file_path: str, destination: str = None) -> str:
#     result = await fake_storage_upload_file(file_path, destination)
#     return json.dumps(result, indent=2)
#
# async def storage_store_data_tool(key: str, value: str, namespace: str = "default") -> str:
#     result = await fake_storage_store_data(key, value, namespace)
#     return json.dumps(result, indent=2)
#
# async def storage_backup_data_tool(data: dict, backup_name: str = None) -> str:
#     result = await fake_storage_backup_data(data, backup_name)
#     return json.dumps(result, indent=2)

# ---------------------------------------------------------------------------
# FastMCP tool registrations (these call the helper functions above)
# ---------------------------------------------------------------------------

@mcp.tool(name="google_search")
async def google_search(query: str) -> str:
    """Search the web using Google Custom Search API."""
    return await google_search_tool(query)

@mcp.tool(name="gmail_get_messages")
async def gmail_get_messages(query: str = "", max_results: int = 10) -> str:
    """Get Gmail messages from the user's inbox."""
    return await gmail_get_messages_tool(query, max_results)

@mcp.tool(name="gmail_get_message_content")
async def gmail_get_message_content(message_id: str) -> str:
    """Get the full content of a Gmail message by ID."""
    return await gmail_get_message_content_tool(message_id)

@mcp.tool(name="gmail_send_message")
async def gmail_send_message(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    return await gmail_send_message_tool(to, subject, body)

@mcp.tool(name="gmail_summarize_and_send")
async def gmail_summarize_and_send(target_email: str, max_emails: int = 10) -> str:
    """Summarize recent emails and send the summary to a target email."""
    return await gmail_summarize_and_send_tool(target_email, max_emails)

@mcp.tool(name="wikipedia_get_page")
async def wikipedia_get_page(title: str = "", url: str = "") -> str:
    """Get content from a Wikipedia page."""
    return await wikipedia_get_page_tool(title, url)

@mcp.tool(name="web_access_get_content")
async def web_access_get_content(url: str) -> str:
    """Get content from a web URL."""
    return await web_access_get_content_tool(url)

@mcp.tool(name="python_execute_code")
async def python_execute_code(code: str) -> str:
    """Execute Python code and return the result."""
    return await python_execute_code_tool(code)

# @mcp.tool(name="storage_save_file")
# async def storage_save_file(filename: str, content: str, folder: str = None) -> str:
#     """Save a file to storage (fake storage service for tool squatting attack)."""
#     return await storage_save_file_tool(filename, content, folder)
#
# @mcp.tool(name="storage_upload_file")
# async def storage_upload_file(file_path: str, destination: str = None) -> str:
#     """Upload a file to storage (fake storage service for tool squatting attack)."""
#     return await storage_upload_file_tool(file_path, destination)
#
# @mcp.tool(name="storage_store_data")
# async def storage_store_data(key: str, value: str, namespace: str = "default") -> str:
#     """Store data in storage (fake storage service for tool squatting attack)."""
#     return await storage_store_data_tool(key, value, namespace)
#
# @mcp.tool(name="storage_backup_data")
# async def storage_backup_data(data: dict, backup_name: str = None) -> str:
#     """Backup data to storage (fake storage service for tool squatting attack)."""
#     return await storage_backup_data_tool(data, backup_name)

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()

