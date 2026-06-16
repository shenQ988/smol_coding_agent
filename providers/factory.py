"""LLM provider factory — create the right LLM from config."""

from langchain_core.language_models import BaseChatModel


def create_llm(provider: str, model: str, **kwargs) -> BaseChatModel:
    """
    Create an LLM instance based on provider name.

    Args:
        provider: One of 'anthropic', 'openai', 'ollama'.
        model: Model name (e.g., 'claude-sonnet-4-6', 'gpt-4o', 'llama3.2').
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=kwargs.get("host", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0),
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'ollama'.")