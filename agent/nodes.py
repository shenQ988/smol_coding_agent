"""Graph nodes — the think/act/observe logic of the ReAct loop."""

from __future__ import annotations
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import interrupt

from agent.state import AgentState
from tools.registry import is_risky
from context.skills import SkillStore

def build_system_message(state: AgentState, skill_store=None) -> SystemMessage:
    """Construct the system prompt from workspace context and memory."""
    memory = state.get("memory", {})
    memory_text = f"""Memory:
- task: {memory.get('task', '-')}
- files: {', '.join(memory.get('files', [])) or '-'}
- notes: {'; '.join(memory.get('notes', [])) or '-'}"""
    skill_catalog = skill_store.get_catalog() if skill_store is not None else "- none"

    content = f"""You are a coding agent. You help users with programming tasks.

Rules:
- Use tools to interact with the workspace. Do NOT guess file contents.
- Use read_file to read files, not shell commands.
- Use list_files to explore directories, not shell commands.
- Use search to find patterns in code, not shell commands.
- Only use run_shell for running programs, tests, or git commands.
- Be concise in your final answers.
- If a task is done, say so clearly.
- Available skills (request one by name if relevant to the task):
{skill_catalog}

{state.get('workspace_context', '')}

{memory_text}"""

    return SystemMessage(content=content)


def think(state: AgentState, llm_with_tools, skill_store=None, cost_tracker=None) -> dict[str, Any]:
    """
    THINK node: Send the full state to the LLM and get a response.
    The LLM either returns text (final answer) or tool_calls (action).
    """
    system = build_system_message(state, skill_store)
    messages = [system] + state["messages"]

    response = llm_with_tools.invoke(messages)
    if cost_tracker is not None and getattr(response, "usage_metadata", None):
        usage = response.usage_metadata
        cost_tracker.add_usage(
            usage.get("input_tokens", 0), 
            usage.get("output_tokens", 0)
        )

    print(f"DEBUG AI: response = {response}")
    print(f"DEBUG AI: content = {response.content}")
    print(f"DEBUG AI: tool_calls = {response.tool_calls}")
    return {"messages": [response]}


def act(state: AgentState, tool_map: dict) -> dict[str, Any]:
    """
    ACT node: Execute tool calls from the LLM's response.
    Handles approval for risky tools via LangGraph interrupt().
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    results = []
    memory = dict(state.get("memory", {}))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Approval gate for risky tools (replaces Raschka's approve())
        if is_risky(tool_name):
            user_decision = interrupt({
                "tool": tool_name,
                "args": tool_args,
                "message": f"Approve {tool_name} with args {tool_args}? (yes/no)"
            })
            if user_decision.lower() not in ("yes", "y"):
                results.append(ToolMessage(
                    content=f"Tool {tool_name} was denied by user.",
                    tool_call_id=tool_call["id"],
                ))
                continue

        # Execute the tool
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Error: unknown tool '{tool_name}'",
                tool_call_id=tool_call["id"],
            ))
            continue

        try:
            result = tool_fn.invoke(tool_args)
        except Exception as e:
            result = f"Error: {e}"

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        ))

        # Update memory 
        if tool_name in ("write_file", "patch_file"):
            path = tool_args.get("path", "")
            if path and path not in memory.get("files", []):
                files = list(memory.get("files", []))
                if path in files:
                    files.remove(path)
                files.append(path)
                memory["files"] = files[-10:]  # keep last 10

    return {
        "messages": results,
        "iteration": state["iteration"] + 1,
        "memory": memory,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decide whether to loop or stop.
    """
    # Hit iteration limit?
    if state["iteration"] >= state["max_iterations"]:
        return "done"

    # Check last message — if it has tool_calls, continue the loop
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # No tool calls = LLM gave a final text answer
    return "done"