#!/usr/bin/env python3
"""
MCP proxy script for connecting Zed to the Context-Engine indexer MCP server.
This script bridges stdio communication (expected by Zed) with HTTP communication.
"""

import asyncio
import json
import sys
import aiohttp
import uuid


async def main():
    """Main proxy function that bridges stdio and HTTP MCP communication."""
    session_id = str(uuid.uuid4())
    base_url = "http://localhost:8003"

    async with aiohttp.ClientSession() as session:
        # Start SSE connection
        async with session.get(f"{base_url}/sse") as response:
            if response.status != 200:
                print(
                    f"Failed to connect to MCP server