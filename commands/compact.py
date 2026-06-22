"""/compact — manually summarize old conversation history."""

from langgraph.graph.message import RemoveMessage
from memory.reduction import compact_history


def run(graph, graph_config, llm, **kwargs) -> str:
    current_state = graph.get_state(graph_config)
    old_messages = current_state.values.get("messages", [])

    if len(old_messages) <= 4:
        return "Not enough history to compact."

    try:
        compacted = compact_history(old_messages, llm)
    except Exception as e:
        return f"Error: compact failed — {e}"

    removals = [RemoveMessage(id=m.id) for m in old_messages]
    graph.update_state(graph_config, {"messages": removals + compacted})

    return f"Compacted {len(old_messages)} messages into {len(compacted)}."