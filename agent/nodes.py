"""Graph nodes — the think/act/observe logic of the ReAct loop."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import interrupt

from agent.state import AgentState
from tools.registry import is_risky
from context.skills import SkillStore

TOOL_TIMEOUT = 30  # seconds before a tool call is killed

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

        # Approval gate for risky tools
        if is_risky(tool_name):
            if tool_name == "run_shell":
                msg = f"Run shell command?\n  $ {tool_args.get('command', '')}"
            else:
                msg = f"Approve {tool_name}?\n  args: {tool_args}"
            user_decision = interrupt({"tool": tool_name, "args": tool_args, "message": msg})
            if user_decision.lower() not in ("yes", "y"):
                results.append(ToolMessage(
                    content=f"Tool {tool_name} was denied by user.",
                    tool_call_id=tool_call["id"],
                ))
                continue

        # Execute the tool with a timeout
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            results.append(ToolMessage(
                content=f"Error: unknown tool '{tool_name}'",
                tool_call_id=tool_call["id"],
            ))
            continue

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool_fn.invoke, tool_args)
                try:
                    result = future.result(timeout=TOOL_TIMEOUT)
                except FuturesTimeoutError:
                    result = f"Error: '{tool_name}' timed out after {TOOL_TIMEOUT}s"
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

    last_tc = last_message.tool_calls[-1] if last_message.tool_calls else None
    last_tool_call = {"name": last_tc["name"], "args": last_tc["args"]} if last_tc else None

    return {
        "messages": results,
        "iteration": state["iteration"] + 1,
        "memory": memory,
        "last_tool_call": last_tool_call,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge after think. Returns one of:
      "continue"  — LLM wants to call a tool, all checks pass
      "done"      — LLM gave a final answer (or empty response)
      "summarize" — hard stop due to iteration cap or loop; ask LLM to wrap up
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    # No tool calls → LLM finished (or returned nothing)
    if not tool_calls:
        if not getattr(last_message, "content", None):
            print("  ⚠ LLM returned an empty response.")
        return "done"

    # LLM wants to keep going — run safety checks before allowing it
    current_call = {"name": tool_calls[0]["name"], "args": tool_calls[0]["args"]}

    if state.get("last_tool_call") == current_call:
        print(f"  ⚠ Loop detected: '{current_call['name']}' repeated with identical args.")
        return "summarize"

    if state["iteration"] >= state["max_iterations"]:
        print(f"  ⚠ Iteration limit ({state['max_iterations']}) reached.")
        return "summarize"

    return "continue"


def summarize(state: AgentState, llm) -> dict[str, Any]:
    """
    Summarize node: invoked when the agent hits a hard stop (iteration cap or loop).
    Asks the LLM — without tools — to tell the user what was done and what remains.
    """
    summary_prompt = SystemMessage(content=(
        "You were stopped before finishing because you hit the step limit or entered a loop. "
        "Review the conversation and give the user a concise summary: "
        "what was completed, what still needs to be done, and any relevant file paths. "
        "Be direct and specific. Do not call any tools."
    ))
    response = llm.invoke([summary_prompt] + state["messages"])
    return {"messages": [response]}