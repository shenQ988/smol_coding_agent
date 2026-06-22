"""Manual context compaction — summarizes old history into one message."""

from langchain_core.messages import SystemMessage, HumanMessage


def summarize_messages(messages: list, llm) -> str:
    """Summarize a list of messages into a plain string using the LLM."""
    transcript_lines = []
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        transcript_lines.append(f"{role}: {content[:1500]}")
    transcript = "\n".join(transcript_lines)
    if len(transcript) > 30000:
        transcript = transcript[:30000] + "\n... (truncated)"

    response = llm.invoke([
        SystemMessage(content=(
            "Summarize this conversation history concisely. Keep: "
            "file names touched, key decisions made, and the current task state. "
            "Drop: routine tool outputs, repeated confirmations."
        )),
        HumanMessage(content=transcript),
    ])
    return response.content


def compact_history(messages: list, llm, keep_recent: int = 4) -> list:
    """Summarize older messages, keep the most recent ones verbatim."""
    if len(messages) <= keep_recent:
        return messages

    recent = messages[-keep_recent:]
    to_summarize = messages[:-keep_recent]

    summary_text = summarize_messages(to_summarize, llm)
    summary_msg = SystemMessage(content=f"[Earlier conversation summary]\n{summary_text}")
    return [summary_msg] + recent