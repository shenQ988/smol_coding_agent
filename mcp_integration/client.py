"""MCP client — connects to MCP servers and discovers tools."""

from __future__ import annotations
import asyncio
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, dict] = {}
        self._stdio_cms: list[Any] = []   # keep context managers alive
        self._session_cms: list[Any] = []

    async def connect(self, name: str, command: str, args: list[str] = None,
                      env: dict[str, str] = None):
        """Connect to one MCP server and discover its tools."""
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        stdio_cm = stdio_client(params)
        read_stream, write_stream = await stdio_cm.__aenter__()
        self._stdio_cms.append(stdio_cm)

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        self._session_cms.append(session_cm)

        await session.initialize()

        result = await session.list_tools()
        for tool in result.tools:
            full_name = f"mcp__{name}__{tool.name}"
            self.tools[full_name] = {
                "name": tool.name,
                "full_name": full_name,
                "server": name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {},
                "session": session,
            }

        self.sessions[name] = session
        print(f"  Connected to {name}: {len(result.tools)} tools discovered")

    async def connect_all(self, servers_config: dict):
        """Connect to all configured MCP servers."""
        for name, config in servers_config.items():
            try:
                await self.connect(
                    name=name,
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env"),
                )
            except Exception as e:
                print(f"  Warning: failed to connect to {name}: {e}")

    async def close(self):
        """Cleanly close all sessions and stdio connections."""
        for cm in reversed(self._session_cms):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        for cm in reversed(self._stdio_cms):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def call_tool(self, full_name: str, args: dict) -> str:
        """Call an MCP tool by its full name."""
        tool_info = self.tools.get(full_name)
        if not tool_info:
            return f"Error: unknown MCP tool '{full_name}'"

        try:
            result = await tool_info["session"].call_tool(
                tool_info["name"], args
            )
            return result.content[0].text if result.content else "(empty result)"
        except Exception as e:
            return f"MCP tool error: {e}"

    def get_all_tools(self) -> list[dict]:
        """Return all discovered tool definitions."""
        return list(self.tools.values())