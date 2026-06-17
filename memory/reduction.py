"""Manual context compaction — summarizes old history into one message."""

from langchain_core.messages import SystemMessage, HumanMessage


def compact_history(messages: list, llm, keep_recent: int = 4) -> list:
    """Summarize older messages, keep the most recent ones verbatim."""
    if len(messages) <= keep_recent:
        return messages   # not enough history to bother compacting

    recent = messages[-keep_recent:]
    to_summarize = messages[:-keep_recent]

    transcript_lines = []
    for m in to_summarize:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        transcript_lines.append(f"{role}: {content[:300]}")
    transcript = "\n".join(transcript_lines)

    summary_response = llm.invoke([
        SystemMessage(content=(
            "Summarize this conversation history concisely. Keep: "
            "file names touched, key decisions made, and the current task state. "
            "Drop: routine tool outputs, repeated confirmations."
        )),
        HumanMessage(content=transcript),
    ])

    summary_msg = SystemMessage(
        content=f"[Earlier conversation summary]\n{summary_response.content}"
    )

    return [summary_msg] + recent