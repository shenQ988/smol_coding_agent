"""Bridge MCP tools to LangChain BaseTool format."""

from __future__ import annotations
import asyncio
from typing import Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model
from mcp.client import MCPManager


def build_pydantic_model(tool_def: dict) -> Type[BaseModel]:
    """Convert MCP JSON Schema to a Pydantic model for LangChain."""
    schema = tool_def.get("input_schema", {})
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_map = {
        "string": str, "integer": int, "number": float,
        "boolean": bool, "array": list, "object": dict,
    }

    fields = {}
    for prop_name, prop_def in properties.items():
        python_type = type_map.get(prop_def.get("type", "string"), str)
        desc = prop_def.get("description", "")

        if prop_name in required:
            fields[prop_name] = (python_type, Field(description=desc))
        else:
            fields[prop_name] = (Optional[python_type], Field(default=None, description=desc))

    if not fields:
        fields["_placeholder"] = (Optional[str], Field(default=None, description="No args needed"))

    safe_name = f"MCP_{tool_def['full_name'].replace('-', '_').replace('.', '_')}"
    return create_model(safe_name, **fields)


class MCPToolWrapper(BaseTool):
    """Wraps an MCP tool as a LangChain BaseTool."""
    name: str = ""
    description: str = ""
    mcp_manager: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs):
        kwargs.pop("_placeholder", None)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._arun(**kwargs))
        finally:
            loop.close()

    async def _arun(self, **kwargs):
        kwargs.pop("_placeholder", None)
        if not self.mcp_manager:
            return "Error: MCP manager not available"
        return await self.mcp_manager.call_tool(self.name, kwargs)


def create_mcp_langchain_tools(mcp_manager: MCPManager) -> list[BaseTool]:
    """Convert all MCP tools to LangChain tools."""
    tools = []
    for tool_def in mcp_manager.get_all_tools():
        try:
            args_schema = build_pydantic_model(tool_def)
        except Exception:
            args_schema = None

        wrapper = MCPToolWrapper(
            name=tool_def["full_name"],
            description=f"[MCP:{tool_def['server']}] {tool_def['description']}",
            mcp_manager=mcp_manager,
        )
        if args_schema:
            wrapper.args_schema = args_schema
        tools.append(wrapper)

    return tools