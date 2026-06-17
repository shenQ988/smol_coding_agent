"""/model — show the current provider and model."""

def run(provider: str = "", model: str = "", **kwargs) -> str:
    return f"Provider: {provider}, Model: {model}"