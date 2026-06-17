#!/usr/bin/env python3
"""CLI entry point — the REPL that drives the agent."""

import argparse
import asyncio
import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from mcp_integration.client import MCPManager
from mcp_integration.tool_bridge import create_mcp_langchain_tools
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import build_graph
from context.workspace import WorkspaceContext
from tools import filesystem, shell, search
from agent.state import create_initial_state
from context.skills import SkillStore
from tools.skills import set_skill_store
from commands.registry import dispatch
from providers.factory import create_llm
from agent.cost_tracker import CostTracker
def _expand_env_vars(obj):
    """Recursively expand ${VAR} references in config strings."""
    if isinstance(obj, str):
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def load_config(path: str = "config.yaml") -> dict:
    """Load config from YAML file, expanding ${VAR} references from environment."""
    load_dotenv()
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return _expand_env_vars(raw)
    return {}


async def main():
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
        await mcp_manager.connect_all(mcp_servers)
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
    

    #create the llm
    llm = create_llm(provider, model, temperature=agent_config.get("temperature", 0))

    cost_tracker = CostTracker(model=model)

    # Build the graph
    graph = build_graph(
        provider=provider,
        model=model,
        max_iterations=max_iter,
        temperature=agent_config.get("temperature", 0),
        extra_tools=mcp_tools, 
        llm = llm, 
        cost_tracker= cost_tracker
    )

    # Session config — LangGraph uses thread_id for session management
    thread_id = "default"
    graph_config = {"configurable": {"thread_id": thread_id}}

    # Welcome message
    print("     (•ᴗ•)  smol — a tiny coding agent   ")
    print("╭────────────────────────────────────────╮")
    print("│           SMOL Coding Agent            │")
    print("│  /help for commands, Ctrl+C to exit    │")
    print(f"│  Provider: {provider}, Model: {model:<16s}│")
    print("╰────────────────────────────────────────╯")
    print()
    command_context = {
        "graph": graph,
        "graph_config": graph_config,
        "llm": llm,      # now this is defined
        "cost_tracker": cost_tracker,
        "provider": provider,
        "model": model,
    }
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
#         if user_input == "/exit":
#             break
#         if user_input == "/help":
#             print("""Commands:
#   /exit    — quit the agent
#   /help    — show this help
#   /model   — show current model
#   /memory  — show agent memory
#   /clear   — clear conversation""")
#             continue
#         if user_input == "/model":
#             print(f"Provider: {provider}, Model: {model}")
#             continue
#         if user_input == "/clear":
#             thread_id = f"session_{id(object())}"
#             graph_config = {"configurable": {"thread_id": thread_id}}
#             print("Conversation cleared.")
#             continue

        if user_input.startswith("/"):
            result = dispatch(user_input, **command_context)

            if result is None:
                print(f"Unknown command: {user_input}")
            elif result == "__EXIT__":
                break
            elif result.startswith("__NEW_THREAD__"):
                thread_id = result.removeprefix("__NEW_THREAD__")
                graph_config = {"configurable": {"thread_id": thread_id}}
                command_context["graph_config"] = graph_config
                print("Conversation cleared.")
            else:
                print(result)
            continue
        

        # Build initial state for this turn
        initial_state = create_initial_state(
            user_message=user_input,
            workspace_context=workspace_prompt,
            max_iterations=max_iter,
        )

        # Run the graph
        try:
            stream_input = initial_state
            while True:
                interrupted = False
                for event in graph.stream(stream_input, graph_config, stream_mode="updates"):
                    if not isinstance(event, dict):
                        continue
                    for node_name, node_output in event.items():
                        if node_name == "__interrupt__":
                            interrupted = True
                            interrupts = node_output if isinstance(node_output, (list, tuple)) else [node_output]
                            for intr in interrupts:
                                msg = intr.value.get("message", f"Approve {intr.value.get('tool')}?") if hasattr(intr, "value") else str(intr)
                                print(f"\n  \033[33m⚠ {msg}\033[0m")
                            try:
                                decision = input("  approve? [y/N] ").strip().lower()
                            except (KeyboardInterrupt, EOFError):
                                decision = "n"
                            stream_input = Command(resume=decision)
                            break  # restart stream with resume command

                        if not isinstance(node_output, dict):
                            continue
                        if node_name == "think":
                            msgs = node_output.get("messages", [])
                            for msg in msgs:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        print(f"  \033[33m⚡ {tc['name']}\033[0m({tc['args']})")
                                elif hasattr(msg, "content") and msg.content:
                                    print(f"\n{msg.content}")
                        elif node_name == "act":
                            msgs = node_output.get("messages", [])
                            for msg in msgs:
                                content = str(msg.content)
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                print(f"  \033[32m→ {content}\033[0m")
                    if interrupted:
                        break  # restart outer while loop with resume
                else:
                    break  # stream exhausted normally, exit while loop

        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

    print("Session ended.")
    await mcp_manager.close()


if __name__ == "__main__":
    asyncio.run(main())