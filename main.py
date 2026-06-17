#!/usr/bin/env python3
"""CLI entry point — the REPL that drives the agent."""

import argparse
import yaml
from pathlib import Path
from mcp.client import MCPManager
from mcp.tool_bridge import create_mcp_langchain_tools
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from context.workspace import WorkspaceContext
from tools import filesystem, shell, search
from agent.state import create_initial_state
from context.skills import SkillStore
from tools.skills import set_skill_store
import asyncio
def load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML file."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="LangGraph Coding Agent")
    parser.add_argument("--provider", default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Load config (CLI args override YAML)
    config = load_config(args.config)
    agent_config = config.get("agent", {})
    mcp_manager = MCPManager()
    mcp_servers = config.get("mcp_servers", {})
    if mcp_servers:
        asyncio.run(mcp_manager.connect_all(mcp_servers))
    if mcp_servers:
        mcp_tools = create_mcp_langchain_tools(mcp_manager)
    else:
        mcp_tools = []

    provider = args.provider or agent_config.get("provider", "anthropic")
    model = args.model or agent_config.get("model", "claude-sonnet-4-6")
    max_iter = args.max_iterations or agent_config.get("max_iterations", 15)

    # Set workspace root for tools
    cwd = args.cwd.resolve()
    filesystem.set_workspace_root(cwd)
    shell.set_workspace_root(cwd)
    search.set_workspace_root(cwd)

    # Scan workspace
    workspace = WorkspaceContext(cwd)
    workspace_prompt = workspace.to_prompt_string()

    # add skills 
    skill_store = SkillStore(Path("./skills"))
    set_skill_store(skill_store)


    # Build the graph
    graph = build_graph(
        provider=provider,
        model=model,
        max_iterations=max_iter,
        temperature=agent_config.get("temperature", 0),
        extra_tools=mcp_tools
    )

    # Session config — LangGraph uses thread_id for session management
    thread_id = "default"
    graph_config = {"configurable": {"thread_id": thread_id}}

    # Welcome message
    print("╭────────────────────────────────────────╮")
    print("│        Coding Agent (LangGraph)        │")
    print("│  /help for commands, Ctrl+C to exit    │")
    print(f"│  Provider: {provider}, Model: {model:<16s}│")
    print("╰────────────────────────────────────────╯")
    print()

    # The REPL loop
    while True:
        try:
            user_input = input("\033[36magent>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input == "/exit":
            break
        if user_input == "/help":
            print("""Commands:
  /exit    — quit the agent
  /help    — show this help
  /model   — show current model
  /memory  — show agent memory
  /clear   — clear conversation""")
            continue
        if user_input == "/model":
            print(f"Provider: {provider}, Model: {model}")
            continue
        if user_input == "/clear":
            thread_id = f"session_{id(object())}"
            graph_config = {"configurable": {"thread_id": thread_id}}
            print("Conversation cleared.")
            continue

        # Build initial state for this turn
        initial_state = create_initial_state(
            user_message=user_input,
            workspace_context=workspace_prompt,
            max_iterations=max_iter,
        )

        # Run the graph
        try:
            # Stream events for visibility
            for event in graph.stream(initial_state, graph_config, stream_mode="updates"):
                if not isinstance(event, dict):
                    print(f"DEBUG: unexpected event type: {type(event)}")
                    continue
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        print(f"DEBUG: unexpected output type from {node_name}: {type(node_output)}")
                        continue
                    print(f"DEBUG: node = {node_name}, keys = {node_output.keys()}")
                    if node_name == "think":
                        # Print the LLM's response
                        msgs = node_output.get("messages", [])
                        for msg in msgs:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"  \033[33m⚡ {tc['name']}\033[0m({tc['args']})")
                            elif hasattr(msg, "content") and msg.content:
                                print(f"\n{msg.content}")

                    elif node_name == "act":
                        # Print tool results
                        msgs = node_output.get("messages", [])
                        for msg in msgs:
                            content = str(msg.content)
                            if len(content) > 500:
                                content = content[:500] + "..."
                            print(f"  \033[32m→ {content}\033[0m")

        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

    print("Session ended.")


if __name__ == "__main__":
    main()